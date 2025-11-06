import json
import pandas as pd
import argparse
import os

from modules.processor import VocabularyProcessor
from modules.processor import PathwayFormatter
from modules.processor import FirstPathwayProcessor
from modules.processor import ViewInformationProcessor
from modules.processor import ReportProcessor
from modules.processor import set_global_vocab_processor
from modules.processor import ConflictDetector
from modules.processor import ConflictResolver
from modules.processor import process_multiple_reports, clean_dataframe_columns, filter_invalid_evidence_associate, PathwayProcessor


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process medical reports and analyze pathways.')
    
    parser.add_argument('--vocab_path', type=str, default='../data_resources/lunguage/Lunguage_vocab.csv',
                        help='Path to vocabulary Excel file')
    parser.add_argument('--dx_pathway_path', type=str, default='../data_resources/dx_pathway.csv',
                        help='Path to diagnosis pathway CSV file')
    parser.add_argument('--output_json_path', type=str, default='./dx_pathway_grouped.json',
                        help='Path to save the grouped pathway JSON file')
    parser.add_argument('--gold_dataset_path', type=str, default='../data_resources/Lunguage_w_prob.csv',
                        help='Path to gold dataset CSV file')
    parser.add_argument('--metadata_path', type=str, default='../data_resources/lunguage/mimic-cxr-2.0.0-metadata.csv',
                        help='Path to MIMIC metadata')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Path to output directory')
    parser.add_argument('--subject_id', type=list, default=[],
                        help='Subject ID to filter data')
    parser.add_argument('--save_output', action='store_true',
                        help='Save output files')
    parser.add_argument('--resolve_conflicts', action='store_true',
                        help='Enable conflict detection and resolution')
    parser.add_argument('--matching', type=str, choices=['model', 'string'], default='model',
                        help='Matching strategy for entity/view/location merges (default: model)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose logging for matching and consolidation')
    parser.add_argument('--level', type=str, choices=['section', 'report'], default='report',
                        help='Processing level - section or report level (default: report)')

    return parser.parse_args()


def main():
    """Main function to process and analyze medical reports."""
    
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    suffix = args.matching if hasattr(args, 'matching') else 'model'
    args.section_output_path = os.path.join(args.output_dir, f'dx_sr_section_level_all_{suffix}.csv')
    args.report_output_path = os.path.join(args.output_dir, f'dx_sr_report_level_all_{suffix}.csv')

    ##########################################################################################
    # 1. Load and preprocess data -> vocab file and pathway file
    vocab_df = pd.read_csv(args.vocab_path)
    dx_pathway_df = pd.read_csv(args.dx_pathway_path)
    
    # 2. Convert to lowercase
    vocab_df = vocab_df.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
    dx_pathway_df = dx_pathway_df.apply(lambda x: x.str.lower() if x.dtype == "object" else x)
    
    # 3. Drop rows with missing disease information
    dx_pathway_df.dropna(subset=['disease', 'specific disease'], inplace=True)
    
    # Initialize processors
    vocab_processor = VocabularyProcessor(vocab_df)
    set_global_vocab_processor(vocab_processor)
    pathway_formatter = PathwayFormatter()
    pathway_processor = FirstPathwayProcessor(pathway_formatter)
    
    # Process pathway data -> turn each pathway into a structured string and store it in the column "formatted_results"
    processed_df = pathway_processor.process_dataframe(dx_pathway_df)
    
    # Create grouped dictionary
    grouped_dict = pathway_processor.create_grouped_dict(processed_df)
    
    # Save to JSON
    with open(args.output_json_path, 'w') as json_file:
        json.dump(grouped_dict, json_file, indent=4)
    
    print(f"Processed {len(processed_df)} rows and saved to {args.output_json_path}")
    print(f"Found {len(pathway_formatter.get_unmatched_terms())} unmatched terms")
    
    # Process view information
    view_processor = ViewInformationProcessor()
    view_merged_df, reviewed_count, merged_count = view_processor.process(
        args.gold_dataset_path,
        args.metadata_path
    )

    if 'status' not in view_merged_df.columns:
        # Convert dx_status and dx_certainty to lowercase
        view_merged_df['dx_status'] = view_merged_df['dx_status'].str.lower()
        view_merged_df['dx_certainty'] = view_merged_df['dx_certainty'].str.lower()
        
        # Create status based on dx_status and dx_certainty
        view_merged_df['status'] = view_merged_df.apply(
            lambda x: (
                ('d' if x['dx_certainty'] == 'definitive' else 't' if x['dx_certainty'] == 'tentative' else '') + 
                ('p' if x['dx_status'] == 'positive' else 'n' if x['dx_status'] == 'negative' else '')
            ),
            axis=1
        )

    if 'idx' not in view_merged_df.columns:
        view_merged_df['idx'] = view_merged_df['ent_idx']

    # Filter by subject ID
    if len(args.subject_id) > 0:
        view_merged_df = view_merged_df[view_merged_df['subject_id'].isin(args.subject_id)]
    
    ##########################################################################################
    # Process reports
    matching_dataframe = view_merged_df.fillna('')
    report_processor = ReportProcessor(matching_dataframe, matching=args.matching, verbose=args.verbose)
    section_level_df, report_level_df = report_processor.process()

    section_level_df = filter_invalid_evidence_associate(matching_dataframe=matching_dataframe, report_level_df=section_level_df)
    report_level_df = filter_invalid_evidence_associate(matching_dataframe=matching_dataframe, report_level_df=report_level_df)
    
    # Process pathways
    pathway_processor = PathwayProcessor(grouped_dict)
    
    # Process based on selected level
    if args.level == 'section':
        section_level_df = pathway_processor.process_dataframe(section_level_df)
        section_level_df.to_csv(f'./output/section_level_df_{suffix}.csv', index=False)
        section_level_df = pd.read_csv(f'./output/section_level_df_{suffix}.csv')
        
        if 'idx' not in section_level_df.columns:
            raise ValueError("idx column not found in section_level_df")
            
        section_level_df_filtered = section_level_df[section_level_df['idx'].notna()]
        study_ids_to_process_section = section_level_df_filtered.study_id.unique()
        output_df = process_multiple_reports(
            section_level_df_filtered, 
            study_ids_to_process_section, 
            report_type='section-level',
            grouped_dict=grouped_dict)
            
        # Select columns for section level output
        output_df = output_df[[
            'idx', 'subject_id', 'study_id', 'sequence', 'section', 'sent', 'ent', 'cat', 'prob',
            'status', 'location', 'evidence', 'associate', 'morphology', 'distribution', 
            'measurement', 'severity', 'comparison', 'onset', 'no change', 'improved', 
            'worsened', 'placement', 'past hx', 'other source', 'assessment limitations', 
            'report', 'view_information', 'pathway'
        ]]
        
        output_path = args.section_output_path
        
    else:  # report level (default)
        report_level_df = pathway_processor.process_dataframe(report_level_df)
        report_level_df.to_csv(f'./output/report_level_df_{suffix}.csv', index=False)
        report_level_df = pd.read_csv(f'./output/report_level_df_{suffix}.csv')
        
        if 'idx' not in report_level_df.columns:
            raise ValueError("idx column not found in report_level_df")
            
        report_level_df_filtered = report_level_df[report_level_df['idx'].notna()]
        study_ids_to_process_report = report_level_df_filtered.study_id.unique()
        output_df = process_multiple_reports(
            report_level_df_filtered,
            study_ids_to_process_report,
            report_type='report-level', 
            grouped_dict=grouped_dict)
            
        # Select columns for report level output
        output_df = output_df[[
            'idx', 'subject_id', 'study_id', 'sequence', 'section', 'sent', 'ent', 'cat', 'prob',
            'status', 'location', 'evidence', 'associate', 'morphology', 'distribution', 
            'measurement', 'severity', 'comparison', 'onset', 'no change', 'improved', 
            'worsened', 'placement', 'past hx', 'other source', 'assessment limitations', 
            'report', 'view_information', 'pathway', 'DxTP'
        ]]
        
        output_path = args.report_output_path

    # Clean output data
    output_df = clean_dataframe_columns(output_df)

    # Save results if requested
    if args.save_output:
        output_df.to_csv(output_path, index=False)
        print(f"Saved output to {output_path}")

    # Print statistics
    print("\n===== FINAL STATISTICS =====")
    print(f"- {args.level} level: {len(output_df)}, "
          f"    - study_id: {output_df.study_id.nunique()}, "
          f"    - subject_id: {output_df.subject_id.nunique()}")

    ##########################################################################################

    # Optional conflict detection and resolution
    if args.resolve_conflicts:
        if args.level == 'section':
            section_level_out = output_df
            report_level_out = pd.read_csv(args.report_output_path) if os.path.exists(args.report_output_path) else pd.DataFrame()
        else:
            report_level_out = output_df
            section_level_out = pd.read_csv(args.section_output_path) if os.path.exists(args.section_output_path) else pd.DataFrame()

        print("\n===== STATUS CONFLICT RESOLUTION =====")
        detector = ConflictDetector(output_dir=args.output_dir)
        
        # Process conflicts even if one level is empty
        section_issue_df, report_issue_df = detector.process_and_save(
            args.section_output_path if not section_level_out.empty else None,
            args.report_output_path if not report_level_out.empty else None
        )

        if not section_issue_df.empty:
            section_issue_df.to_csv(os.path.join(args.output_dir, f'section_issue_{suffix}.csv'), index=False)
        if not report_issue_df.empty:
            report_issue_df.to_csv(os.path.join(args.output_dir, f'report_issue_{suffix}.csv'), index=False)
        
        resolver = ConflictResolver(output_dir=args.output_dir)
        final_section_level, final_report_level = resolver.process_conflicts(
            section_level_out if not section_level_out.empty else None,
            report_level_out if not report_level_out.empty else None,
            section_issue_df if not section_issue_df.empty else None,
            report_issue_df if not report_issue_df.empty else None,
            save_output=args.save_output)

        # Reassign idx values and clean references for the level being processed
        print("Processing final outputs...")
        temp_proc = ReportProcessor(output_df)
        
        if args.level == 'report' and not report_level_out.empty:
            final_report_level = temp_proc._reassign_idx_with_evidence_associate_update(final_report_level)
            final_report_level = temp_proc._clean_invalid_references(final_report_level)
            output_df = final_report_level
        elif args.level == 'section' and not section_level_out.empty:
            final_section_level = temp_proc._reassign_idx_with_evidence_associate_update(final_section_level)
            final_section_level = temp_proc._clean_invalid_references(final_section_level)
            output_df = final_section_level
        
        # Save the final cleaned data
        if args.save_output and not output_df.empty:
            output_df.to_csv(f'./output/final_{args.level}_level_{suffix}.csv', index=False)
            print(f"Saved final cleaned output (suffix={suffix})")
    
        # Print results
        print(f"Final {args.level} level rows: {len(output_df)}")


if __name__ == "__main__":
    main()