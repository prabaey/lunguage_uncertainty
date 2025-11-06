import os
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set, Union
import re
from tqdm import tqdm
import Levenshtein
import pandas as pd
import numpy as np

# Constants
from .constants import (
    VALID_ATTR_CATEGORY2, REPORT_LEVEL_CATEGORY, VALID_ATTR_CATEGORY,
    CLINICAL_ATTR_CAT, ATTRIBUTE_NOT_ALLOWED_SUBCATEGORY, ATTRIBUTE_NOT_ALLOWED_CATEGORY
)

global_vocab_processor = None


def row_to_string(row: pd.Series) -> str:
    """Convert a row (Series) to a string, ignoring NaN values."""
    return ', '.join([f"{col}:{str(val)}" for col, val in row.items() if pd.notna(val)])


def process_single_report(report_df: pd.DataFrame, 
                        report_type: str = 'section-level', 
                        grouped_dict: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Process a single report by iteratively applying pathways until convergence.
    
    Args:
        report_df: DataFrame containing the report data
        report_type: Type of report ('section-level' or other)
        grouped_dict: Dictionary containing grouped pathway data
        
    Returns:
        Updated DataFrame with processed pathways
    """
    prev_len = 0
    test_num = 0
    current_len = len(report_df)
    previous_further_match_idx: Set[Any] = set()
    
    # Create StatusAnalyzer for model-based similarity
    status_analyzer = StatusAnalyzer(verbose=False, matching='model')
    study_processor = StudyProcessor(status_analyzer)
    pathway_processor = PathwayProcessor(grouped_dict)
    
    # keep going while new rows are being added
    while current_len != prev_len:
        prev_len = current_len
        test_num += 1

        if report_df.idx.notna().all():
            # invoke pathways, append new rows into report_df
            # note: report_df has column "pathway", which indicates what pathway to use
            # previous_further_match_idx indicates which indices' pathways were already expanded, to avoid doing these again
            report_df, _ = study_processor.process_study(report_df, list(previous_further_match_idx), report_type)
        else:
            break
        
        # find further matches for all newly generated rows
        further_match = report_df[report_df['sent'].isna()].copy()

        # fill in pathway column for further matches, based on view and entity name
        # we do this in the same way as we did before for the original rows
        further_match['pathway'] = further_match.apply(
            lambda x: pathway_processor.get_pathway_for_ent_and_view(x['ent'], x['view_information'], x),
            axis=1
        )
        
        # select all rows for which a pathway was found
        pathway_match = further_match[further_match['pathway'].notna()]
        previous_further_match_idx.update(report_df[report_df['pathway'].notna()]['idx']) # update list of previously processed indices
                                                                                          # to avoid processing the same thing twice
        
        # update pathway column in report_df for all new rows for which a pathway was found
        pathway_mapping = pathway_match.set_index('idx')['pathway'].to_dict()
        report_df['pathway'] = report_df['idx'].map(pathway_mapping).fillna(report_df['pathway'])
                
        current_len = len(report_df)
    
    # Note: idx reassignment is now handled only at the initial data creation stage
    # for both section-level and report-level to avoid redundant processing
    
    return report_df


def process_multiple_reports(dataframe: pd.DataFrame, 
                            study_ids: List[str], 
                            report_type: str = 'section-level', 
                            grouped_dict: Dict[str, Any] = None) -> pd.DataFrame:
    """
    Process multiple reports by study ID.
    
    Args:
        dataframe: DataFrame containing all reports
        study_ids: List of study IDs to process
        report_type: Type of report ('section-level' or other)
        grouped_dict: Dictionary containing grouped pathway data
        
    Returns:
        Combined DataFrame with all processed reports
    """
    final_results = pd.DataFrame()

    # process each study separately
    for study_id in study_ids:
        current_std = dataframe[dataframe['study_id'] == study_id].copy()
        updated_report = process_single_report(current_std, report_type, grouped_dict)
        final_results = pd.concat([final_results, updated_report], ignore_index=True)

    return final_results


def targeting_studies(section_level_df: pd.DataFrame, report_level_df: pd.DataFrame, 
                     pathway_count: pd.DataFrame, max_display: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter and display information about targeted studies.
    
    Args:
        section_level_df: Section-level DataFrame
        report_level_df: Report-level DataFrame
        pathway_count: DataFrame with pathway counts by study
        max_display: Maximum number of studies to display
        
    Returns:
        Tuple of filtered (section_level_df, report_level_df)
    """
    section_level_df = section_level_df[section_level_df['study_id'].isin(pathway_count['study_id'])]
    report_level_df = report_level_df[report_level_df['study_id'].isin(pathway_count['study_id'])]
    
    for count, study_id in enumerate(pathway_count['study_id'].unique(), start=1):
        if count >= max_display:
            break
        
        cur_section_std = section_level_df[section_level_df['study_id'] == study_id]
        cur_report_std = report_level_df[report_level_df['study_id'] == study_id]
        
        print(f"Section Level - dx: {cur_section_std[cur_section_std['pathway'].notna()].ent.values}")
        print(f"Report Level - dx: {cur_report_std[cur_report_std['pathway'].notna()].ent.values}")
            
    return section_level_df, report_level_df


def clean_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean text in DataFrame columns.
    
    Args:
        df: DataFrame to clean
        
    Returns:
        Cleaned DataFrame
    """
    for column in df.select_dtypes(include=['object']).columns:
        if column not in ['report', 'sent']:
            df[column] = df[column].apply(TextProcessor.clean_text)

    return df

def set_global_vocab_processor(vocab_processor):
    global global_vocab_processor
    global_vocab_processor = vocab_processor


def row_to_string(row: pd.Series) -> str:
    """Convert a row (Series) to a string, ignoring NaN values."""
    return ', '.join([f"{col}:{str(val)}" for col, val in row.items() if pd.notna(val)])


class VocabularyProcessor:
    """Processes medical vocabulary and maps terms to categories."""
    
    def __init__(self, vocab_df: pd.DataFrame, 
                 fuzzy_max_distance: int = 2, fuzzy_min_jaccard: float = 0.7):
        """
        Initialize the VocabularyProcessor.
        
        Args:
            vocab_df: DataFrame containing vocabulary data
        """
        self.vocab_df = vocab_df
        self.fuzzy_max_distance = fuzzy_max_distance
        self.fuzzy_min_jaccard = fuzzy_min_jaccard
        # Valid categories for observations, locations, and attributes
        self.valid_obs_categories = ['oth', 'patient info.', 'cof', 'cf', 'ncd', 'pf']
        self.valid_loc_category = ['location']
        self.valid_attr_category = ['placement', 'other source', 'assessment limitations', 'past hx', 
                                   'morphology', 'distribution', 'measurement', 'severity', 'comparison', 
                                   'onset', 'improved', 'no change', 'worsened']
        
    def get_clean_categories(self, term: str) -> List[str]:
        """
        Get clean categories for a term from vocabulary.
        
        Args:
            term: Term to look up in vocabulary
            
        Returns:
            List of category strings
        """
        category_info = self.vocab_df.loc[
            # (self.vocab_df['raw_term'] == term) | 
            (self.vocab_df['target_term'] == term) | 
            (self.vocab_df['normed_term'] == term), ['category', 'subcategory']
        ].drop_duplicates()
        
        # Convert to list of formatted category strings
        categories = category_info.apply(
            lambda x: f"{x['category']} ({x['subcategory']})".lower() if pd.notna(x['subcategory']) else x['category'].lower(), 
            axis=1
        ).values.tolist()
        
        return categories


    def _best_vocab_match(self, term: str) -> Optional[pd.Series]:
        """
        Find the best matching vocabulary row using Levenshtein distance and Jaccard similarity
        when exact matches fail.
        """
        if not term or not isinstance(term, str):
            return None
        term_l = term.lower().strip()
        candidates = self.vocab_df[['target_term', 'normed_term']].fillna('')
        best_row_idx = None
        best_score = (-1.0, float('inf'))  # (jaccard, distance) higher jaccard, lower distance preferred

        def jaccard(a: str, b: str) -> float:
            sa, sb = set(a.split()), set(b.split())
            return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0

        for idx, row in candidates.iterrows():
            for col in ['target_term', 'normed_term']:
                val = str(row[col]).lower().strip()
                if not val:
                    continue
                dist = Levenshtein.distance(val, term_l)
                jac = jaccard(val, term_l)
                # Filter by thresholds
                if jac >= self.fuzzy_min_jaccard or dist <= self.fuzzy_max_distance:
                    score = (jac, dist)
                    if (score[0] > best_score[0]) or (score[0] == best_score[0] and score[1] < best_score[1]):
                        best_score = score
                        best_row_idx = idx
        if best_row_idx is None:
            return None
        return self.vocab_df.loc[best_row_idx]

    def get_clean_categories_fuzzy(self, term: str) -> List[str]:
        """
        Fallback: try fuzzy match against vocabulary and return categories if found.
        """
        best_row = self._best_vocab_match(term)
        if best_row is None:
            return []
        # Prefer categories from the best row
        category = best_row.get('category')
        subcategory = best_row.get('subcategory')
        formatted = []
        if pd.notna(category) and category != '':
            if pd.notna(subcategory) and subcategory != '':
                formatted.append(f"{str(category).lower()} ({str(subcategory)})")
            else:
                formatted.append(str(category).lower())
        return formatted
    
    def find_queries(self, specific_disease: str, super_disease: Optional[str] = None) -> List[str]:
        """
        Find queries based on disease and specific disease from vocabulary.
        
        Args:
            specific_disease: Specific disease term
            super_disease: Parent disease term (optional)
            
        Returns:
            List of query terms
        """
        # Search for normed_term from vocab
        normed_term = []
        
        if pd.notna(specific_disease):
            normed_term = self.vocab_df.loc[
                (self.vocab_df['target_term'] == specific_disease.lower()), 'normed_term'
            ].unique()
            
            if pd.isna(normed_term).all() and pd.notna(super_disease):
                normed_term = self.vocab_df.loc[
                    (self.vocab_df['target_term'] == super_disease.lower()), 'normed_term'
                ].unique()
            elif (normed_term.size == 0 if isinstance(normed_term, np.ndarray) else pd.isna(normed_term)) and \
                (super_disease.size == 0 if isinstance(super_disease, np.ndarray) else pd.isna(super_disease)):
                normed_term = []
        
        if len(normed_term) == 0:
            return []
        
        # Get all target_terms associated with the found normed_term
        queries = self.vocab_df[self.vocab_df['normed_term'].isin(normed_term)]['target_term'].unique().tolist()
        
        return queries
    
    def find_cat(self, disease: str) -> Tuple[List[str], List[str]]:
        """
        Find category and subcategory for a disease term.
        
        Args:
            disease: Disease term to look up
            
        Returns:
            Tuple of (category list, subcategory list)
        """
        category, subcategory = [], []
        
        if pd.notna(disease):
            # Try to find category by normed_term
            category = self.vocab_df.loc[
                (self.vocab_df['normed_term'] == disease.lower()), 'category'
            ].unique().tolist()
            
            # If not found, try by target_term
            if len(category) == 0:
                category = self.vocab_df.loc[
                    (self.vocab_df['target_term'] == disease.lower()), 'category'
                ].unique().tolist()
        
            # Try to find subcategory by normed_term
            subcategory = self.vocab_df.loc[
                (self.vocab_df['normed_term'] == disease.lower()), 'subcategory'
            ].unique().tolist()
            
            # If not found, try by target_term
            if len(subcategory) == 0:
                subcategory = self.vocab_df.loc[
                    (self.vocab_df['target_term'] == disease.lower()), 'subcategory'
                ].unique().tolist()
        
        return category, subcategory


class PathwayFormatter:
    """Formats pathway information from medical data."""
    
    def __init__(self):
        """
        Initialize the PathwayFormatter.
        """
        global global_vocab_processor
        self.columns_to_combine = [
            'view', 'observation 1', 'status 1', 'loc 1', 'attributes 1-1', 'attributes 1-2', 'attributes 1-3',
            'observation 2', 'status 2', 'loc 2', 'attributes 2-1', 'attributes 2-2',
            'observation 3', 'status 3', 'loc 3', 'attributes 3-1', 
            'observation 4', 'status 4', 'loc 4', 'attributes 4-1'
        ]
        self.term_list = []
        
    def format_row(self, row: pd.Series) -> str:
        """
        Format a row of pathway data.
        
        Args:
            row: DataFrame row containing pathway data
            
        Returns:
            Formatted pathway string
        """
        result = []
        current_group = []
        previous_number = None
        matching_results = {
            'success': [],
            'failure': []
        }

        # combine all columns in the pathway file into one machine-readable string
        for col in self.columns_to_combine:
            if pd.notna(row[col]):
                col_parts = col.split(' ')
                col_type = col_parts[0]
                col_number = col_parts[1].split('-')[0] if len(col_parts) > 1 else None

                if previous_number is not None and previous_number != col_number:
                    result.append(' > '.join(current_group))
                    current_group = []

                term = row[col]
                col_name = 'obs' if col_type == 'observation' else 'loc' if col_type == 'loc' else 'attr' if col_type == 'attributes' else 'status' if col_type == 'status' else 'view'

                if col_name in ['status', 'view']:
                    current_group.append(f"{col_name}: {term}")
                    previous_number = col_number
                    continue
                
                # map term to category (e.g. location, morphology, improved, etc) using vocabulary
                categories_with_sub = global_vocab_processor.get_clean_categories(term)

                if len(categories_with_sub) == 0:
                    # Try fuzzy fallback matching using vocab Levenshtein/Jaccard
                    categories_with_sub = global_vocab_processor.get_clean_categories_fuzzy(term)
                    if len(categories_with_sub) == 0:
                        valid_categories = ['Nomatch']
                        matching_results['failure'].append((term, col_name, 'No categories found'))
                        self.term_list.append(term)
                    else:
                        valid_categories = categories_with_sub
                else:
                    clean_categories = [cat.split('(')[0].strip() for cat in categories_with_sub]
                    
                    try:
                        # only accept categories that are allowed for a particular type of column
                        if col_name == 'obs':
                            valid_categories = [cat_original for cat_original, cat_clean 
                                             in zip(categories_with_sub, clean_categories)
                                             if cat_clean in global_vocab_processor.valid_obs_categories]
                        elif col_name == 'loc':
                            valid_categories = [cat_original for cat_original, cat_clean 
                                             in zip(categories_with_sub, clean_categories)
                                             if cat_clean == 'location']
                        else:  # attributes
                            valid_categories = [cat_original for cat_original, cat_clean 
                                             in zip(categories_with_sub, clean_categories)
                                            if any(voc.lower() == cat_clean for voc in global_vocab_processor.valid_attr_category)]
                        
                        if valid_categories:
                            matching_results['success'].append((term, col_name, categories_with_sub))
                        else:
                            matching_results['failure'].append((term, col_name, categories_with_sub))
                            valid_categories = ['Nomatch']
                            self.term_list.append(term)
                    except Exception as e:
                        matching_results['failure'].append((term, col_name, str(e)))
                        valid_categories = ['Error']
                        self.term_list.append(term)

                current_group.append(f"{col_name}: {term} ({'/'.join(valid_categories)})")
                previous_number = col_number
        
        if current_group:
            result.append(' > '.join(current_group))

        # Print failures if any
        if matching_results['failure']:
            print("\nFailed to match terms:")
            for term, col_name, reason in matching_results['failure']:
                print(f"✗ {col_name} '{term}' failed: {reason}")
            input("Matching fail exists. STOP!")
        # return machine-readable string for pathway
        return ' && '.join(result)
    
    def get_unmatched_terms(self) -> List[str]:
        """
        Get list of terms that couldn't be matched.
        
        Returns:
            List of unmatched terms
        """
        return self.term_list


class FirstPathwayProcessor:
    """Processes and groups pathway data."""
    
    def __init__(self, pathway_formatter: PathwayFormatter):
        """
        Initialize the PathwayProcessor.
        
        Args:
            pathway_formatter: PathwayFormatter instance
        """
        global global_vocab_processor
        self.pathway_formatter = pathway_formatter
        
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a dataframe to add formatted pathway information.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with added 'formatted_results' column
        """
        result_df = df.copy()
        result_df['formatted_results'] = result_df.apply(self.pathway_formatter.format_row, axis=1)
        return result_df
    
    def create_grouped_dict(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Group pathway data by disease and specific disease.
        
        Args:
            df: DataFrame containing pathway data
            
        Returns:
            Dictionary with grouped pathway data
        """
        grouped_dict = {}
        
        for _, row in df.iterrows():
            disease = row['disease']
            specific_disease = row['specific disease']
            
            if specific_disease not in grouped_dict:
                ent_cat, ent_subcat = global_vocab_processor.find_cat(specific_disease)
                grouped_dict[specific_disease] = {
                    'queries': global_vocab_processor.find_queries(specific_disease, disease),
                    'ent_cat': ent_cat,
                    'ent_subcat': ent_subcat,
                    'pathway': [row['formatted_results']]  # Initialize with a list
                }
            else:
                # Append to the list of pathways if the specific disease already exists
                grouped_dict[specific_disease]['pathway'].append(row['formatted_results'])
                
        return grouped_dict

class DataFrameProcessor:
    def __init__(self, df):
        self.df = df.copy()
    
    def process_text_columns(self, exclude_columns=None):
        """Process all string columns with text cleaning."""
        exclude_columns = exclude_columns or []
        
        for column in self.df.select_dtypes(include=['object']).columns:
            if column not in exclude_columns:
                # Get sample of values that will be changed
                mask = self.df[column].astype(str).str.match(r'.*(\s{2,}|^,|,$|,\s*,).*')
                examples = self.df[mask][column].head()
                
                # Clean the column
                self.df[column] = self.df[column].apply(TextProcessor.clean_text)
                
        return self
    
    def apply_term_mappings(self, column_mappings):
        """Apply multiple term mappings to specified columns."""
        for column, mapping in column_mappings.items():
            if column in self.df.columns:
                self.df[column] = (
                    self.df[column]
                    .replace(mapping)
                    .str.replace('Performed Desc', '', regex=False)
                )
        return self
    
    def apply_mapping_to_column(self, column_name, mapping):
        """Apply text cleaning, term replacement, and deduplication to a column."""
        self.df[column_name] = self.df[column_name].apply(
            lambda x: TextProcessor.remove_duplicates_and_sort(
                TextProcessor.replace_terms(
                    TextProcessor.clean_text(str(x)), mapping
                )
            )
        )
        return self
    
    def get_unique_mapped_terms(self, column_name, mapping):
        """Get unique terms from a column after applying mappings."""
        # Create a copy to avoid modifying the original
        temp_df = self.df.copy()
        
        # Apply mapping
        temp_df[column_name] = temp_df[column_name].apply(
            lambda x: TextProcessor.remove_duplicates_and_sort(
                TextProcessor.replace_terms(
                    TextProcessor.clean_text(str(x)), mapping
                )
            )
        )
        
        # Collect all unique terms
        unique_terms = set()
        temp_df[column_name].apply(lambda x: unique_terms.update(x.split(', ')))
        return unique_terms
    
    def create_view_information(self, columns_to_check):
        """Create a consolidated view information column from multiple columns."""
        def process_row(row):
            values = []
            for col in columns_to_check:
                if col in row:
                    val = row[col]
                    # Handle different types of values
                    if val is None or (isinstance(val, float) and pd.isna(val)):
                        continue
                    elif isinstance(val, list):
                        # Filter out None/NaN values from list
                        valid_vals = [v for v in val if v is not None and not (isinstance(v, float) and pd.isna(v))]
                        values.extend(valid_vals)
                    else:
                        values.append(val)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_values = []
            for val in values:
                if val not in seen:
                    seen.add(val)
                    unique_values.append(val)
            
            return ', '.join(str(v) for v in unique_values)
        
        self.df['view_information'] = self.df.apply(process_row, axis=1)
        return self
    
    def get_dataframe(self):
        """Return the processed dataframe."""
        return self.df

class ViewInformationProcessor:
    def __init__(self):
        # Define mappings
        self.performed_mapping = {
            'CHEST (PORTABLE AP)': 'AP',
            'CHEST (PA AND LAT)': 'PA, LATERAL',
            'CHEST (PA AND LAT) PORT': 'PA, LATERAL',
            'ABDOMEN (SUPINE AND ERECT) PORT': 'SUPINE, ERECT'
        }
        
        self.term_mapping = {
            'antero-posterior': 'AP',
            'postero-anterior': 'PA',
            'left lateral': 'LL'
        }
        
        self.view_mapping = {
            'AP': ['antero-posterior'],
            'PA': ['postero-anterior'],
            'LL': ['left lateral'],
            'LATERAL': ['lateral'],
            'ERECT': ['Erect'],
        }
        
        self.columns_to_check = [
            'ViewPosition', 'PerformedProcedureStepDescription', 
            'ProcedureCodeSequence_CodeMeaning', 'ViewCodeSequence_CodeMeaning', 
            'PatientOrientationCodeSequence_CodeMeaning'
        ]
    
    def process(self, reviewed_df_path, metadata_path):
        """Process the data files and return the merged result."""
        # Load data
        metadata_df = pd.read_csv(metadata_path)

        # Add 'p' prefix to subject_id and 's' prefix to study_id
        metadata_df['subject_id'] = 'p' + metadata_df['subject_id'].astype(str).str.lstrip('p')
        metadata_df['study_id'] = 's' + metadata_df['study_id'].astype(str)

        unique_studies = metadata_df.drop_duplicates('study_id').set_index('study_id')

        # Get aggregated view information for specific columns that need list aggregation 
        view_aggs = (metadata_df
        .groupby('study_id')
        .agg({
            'PerformedProcedureStepDescription': lambda x: list(x.unique()),
            'ViewPosition': lambda x: list(x.unique()),
            'ViewCodeSequence_CodeMeaning': lambda x: list(x.unique()),
            'PatientOrientationCodeSequence_CodeMeaning': lambda x: list(x.unique())
        })
        )

        # Update the specific columns in unique_studies with the aggregated lists
        unique_studies.update(view_aggs)

        seq_meta = unique_studies.reset_index().sort_values(by=['subject_id', 'StudyDate', 'StudyTime'])
        seq_meta['temp_sequence'] = seq_meta.groupby('subject_id').cumcount() + 1

        seq_meta['sequence'] = seq_meta.groupby(['subject_id', 'study_id'])['temp_sequence'].transform('first')

        reviewed_df = pd.read_csv(reviewed_df_path)
        
        # Process reviewed_df
        reviewed_processor = DataFrameProcessor(reviewed_df)
        processed_reviewed_df = reviewed_processor.process_text_columns(
            exclude_columns=['report', 'sent']
        ).get_dataframe()
        
        # Process over_1to15
        over_processor = DataFrameProcessor(seq_meta)
        column_mappings = {
            'PerformedProcedureStepDescription': self.performed_mapping,
            'ProcedureCodeSequence_CodeMeaning': self.performed_mapping,
            'ViewCodeSequence_CodeMeaning': self.term_mapping
        }
        
        processed_over_df = over_processor.apply_term_mappings(
            column_mappings
        ).create_view_information(
            self.columns_to_check
        ).get_dataframe()
        
        # # Get unique mapped terms (for analysis if needed)
        # unique_terms = over_processor.get_unique_mapped_terms(
        #     'view_information', self.view_mapping
        # )
        
        # Merge dataframes
        view_merged_df = processed_reviewed_df.merge(
            processed_over_df[['subject_id', 'study_id', 'sequence', 'view_information']], 
            how='left', 
            on=['subject_id', 'study_id', 'sequence']
        )
        view_merged_df = view_merged_df.drop_duplicates()
        
        return view_merged_df, len(processed_reviewed_df), len(view_merged_df)

class ReportProcessor:
    """Processes medical reports at section and report levels."""
    
    def __init__(self, matching_dataframe, matching: str = 'model', verbose: bool = False):
        self.matching_dataframe = matching_dataframe
        self.matching = matching
        self.verbose = verbose
        self.status_processor = StatusAnalyzer(verbose=verbose, matching=matching)
    
    def create_section_level_df(self):
        """Create section level dataframe from matching_dataframe."""
        section_level_df_list = []
        grouped = self.matching_dataframe.groupby('study_id')
        
        for study_id, group in tqdm(grouped, desc="Processing subjects for section level"):
            # Keep HIST sections as is
            hist_rows = group[group['section'] == 'hist']
            section_level_df_list.append(hist_rows)
            
            # Process FIND and IMPR sections separately
            for section in ['find', 'impr']:
                section_rows = group[group['section'] == section]
                
                # find duplicate rows in each report section, and merge them as needed
                # solve status conflicts according to rules we set
                processed_section = self.status_processor.process_subgroups(section_rows)
                
                if not processed_section.empty:
                    section_level_df_list.append(processed_section)
        
        section_level_df = pd.concat(section_level_df_list, ignore_index=True) if section_level_df_list else pd.DataFrame()
        
        # ✅ CRITICAL: Add idx reassignment BEFORE pathway expansion (like report-level)
        section_level_df = self._reassign_idx_with_evidence_associate_update(section_level_df)
        
        # Log summary for specific study
        # if 's59542064' in grouped.groups:
        #     s59542064_data = section_level_df[section_level_df['study_id'] == 's59542064']
        #     print(f"\n📋 s59542064 Section-level Summary:")
        #     print(f"Final rows: {len(s59542064_data)}")
        #     print(f"Entities: {s59542064_data['ent'].unique()}")
        #     print(f"Statuses: {s59542064_data['status'].unique()}")
        #     input("Press Enter to continue after s59542064 section-level processing...")
        
        return section_level_df    
        
    def create_report_level_df(self):
        """Create report level dataframe from matching_dataframe."""
        report_level_df_list = []
        grouped = self.matching_dataframe.groupby('study_id')
        
        for study_id, group in tqdm(grouped, desc="Processing subjects for report level"):
            # Keep HIST sections as is
            hist_rows = group[group['section'] == 'hist']
            report_level_df_list.append(hist_rows)
            
            # Combine FIND and IMPR sections
            # find duplicate rows in full report (FIND and IMPR), and merge them as needed
            # solve status conflicts according to rules we set
            find_impr_rows = group[group['section'].isin(['find', 'impr'])]
            processed_rows = self.status_processor.process_subgroups(find_impr_rows)
            
            if not processed_rows.empty:
                report_level_df_list.append(processed_rows)
        
        if not report_level_df_list:
            return pd.DataFrame()
            
        report_level_df = pd.concat(report_level_df_list, ignore_index=True)
        
        # Add diagnosis type column
        # store whether it's a past diagnosis (from history section, or pastHx relation) or a current diagnosis (anything else)
        report_level_df['DxTP'] = report_level_df.apply(
            lambda row: 'past dx' if 'hist' in row['section'] or 
                      (pd.notna(row['past hx']) and row['past hx'].strip() != '') 
                      else 'current dx', axis=1
        )
        
        # CRITICAL: Reassign idx values BEFORE pathway expansion to ensure correct references
        # This prevents newly generated rows from referencing wrong idx values
        report_level_df = self._reassign_idx_with_evidence_associate_update(report_level_df)
        
        # self.status_processor.print_merge_summary()
        return report_level_df
    
    def _reassign_idx_with_evidence_associate_update(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reassign idx values per study_id and update evidence/associate references.
        
        Args:
            df: DataFrame with potential idx duplications across sections
            
        Returns:
            DataFrame with reassigned idx values and updated evidence/associate references
        """
        if df.empty:
            return df
            
        print("\n=== IDX REASSIGNMENT DEBUG ===")
        print(f"Total rows to process: {len(df)}")
        
        # Check original duplicates
        original_duplicates = df.groupby(['study_id', 'idx']).size()
        original_dup_counts = original_duplicates[original_duplicates > 1]
        print(f"Original duplicate idx pairs: {len(original_dup_counts)}")
        
        result_dfs = []
        total_updates = {'evidence': 0, 'associate': 0}
        processed_studies = 0
        
        for study_id, group in df.groupby('study_id'):
            group = group.copy()
            
            # Create mapping from old idx to new idx
            # Strategy: Keep first occurrence of each idx, reassign duplicates
            old_indices = group['idx'].tolist()
            new_indices = []
            used_indices = set()
            new_idx_counter = 1
            
            # Process each position individually
            for i, old_idx in enumerate(old_indices):
                # Check if this is the first occurrence of this old_idx
                first_occurrence = old_idx not in old_indices[:i]
                
                if first_occurrence:
                    # First occurrence - keep original idx if available
                    if old_idx not in used_indices:
                        new_indices.append(old_idx)
                        used_indices.add(old_idx)
                    else:
                        # Original idx is already used, find next available
                        while new_idx_counter in used_indices:
                            new_idx_counter += 1
                        new_indices.append(new_idx_counter)
                        used_indices.add(new_idx_counter)
                        new_idx_counter += 1
                else:
                    # This is a duplicate - find next available idx
                    while new_idx_counter in used_indices:
                        new_idx_counter += 1
                    new_indices.append(new_idx_counter)
                    used_indices.add(new_idx_counter)
                    new_idx_counter += 1
            
            # Create mapping for evidence/associate updates
            # Map each old_idx to its corresponding new_idx
            idx_mapping = {}
            for old_idx, new_idx in zip(old_indices, new_indices):
                idx_mapping[old_idx] = new_idx
            
            # Only show detailed debug for first few studies or if there are significant changes
            show_details = processed_studies < 3 or len(original_dup_counts) > 0
            
            # if show_details:
            #     print(f"\n--- Processing study_id: {study_id} ---")
            #     print(f"  Rows: {len(group)}")
            #     print(f"  Original idx: {old_indices}")
            #     print(f"  New idx: {new_indices}")
            #     print(f"  Mapping: {idx_mapping}")
            
            # Update idx values
            group['idx'] = new_indices
            
            # Update evidence and associate fields
            for col in ['evidence', 'associate']:
                if col in group.columns:
                    non_empty_rows = group[group[col].notna() & (group[col].astype(str).str.strip() != '')]
                    if len(non_empty_rows) > 0:
                        # if show_details:
                        #     print(f"\n  {col.upper()} field updates:")
                        
                        # Build validation maps for this group
                        idx_to_ent = {}
                        ent_to_primary_idx = {}
                        try:
                            idx_to_ent = group[['idx', 'ent']].dropna().drop_duplicates().set_index('idx')['ent'].to_dict()
                            ent_to_primary_idx = (
                                group[['ent', 'idx']]
                                .dropna()
                                .groupby('ent', as_index=False)['idx']
                                .min()
                                .set_index('ent')['idx']
                                .to_dict()
                            )
                        except Exception:
                            pass

                        for idx, row in non_empty_rows.iterrows():
                            original_value = row[col]
                            updated_value = self._update_idx_references_in_field(
                                original_value,
                                idx_mapping,
                                idx_to_ent=idx_to_ent,
                                ent_to_primary_idx=ent_to_primary_idx
                            )
                            if original_value != updated_value:
                                # if show_details:
                                #     print(f"    Row {idx}: '{original_value}' → '{updated_value}'")
                                total_updates[col] += 1
                                # Update the actual value in the group
                                group.loc[idx, col] = updated_value
                            else:
                                pass
                                # if show_details:
                                #     print(f"    Row {idx}: '{original_value}' (no change)")
            
            
            result_dfs.append(group)
            processed_studies += 1
        
        result_df = pd.concat(result_dfs, ignore_index=True)
        
        # Check final duplicates
        final_duplicates = result_df.groupby(['study_id', 'idx']).size()
        final_dup_counts = final_duplicates[final_duplicates > 1]
        print(f"\n=== FINAL RESULTS ===")
        print(f"Processed {processed_studies} study_ids")
        print(f"Final duplicate idx pairs: {len(final_dup_counts)}")
        print(f"Total evidence updates: {total_updates['evidence']}")
        print(f"Total associate updates: {total_updates['associate']}")
        print("=== END IDX REASSIGNMENT DEBUG ===\n")
        
        return result_df
    
    def _update_idx_references_in_field(self, field_value: str, idx_mapping: Dict[int, int], idx_to_ent: Dict[int, str] = None, ent_to_primary_idx: Dict[str, int] = None) -> str:
        """
        Update idx references in evidence/associate fields.
        Remove references to idx values that don't exist in the mapping.
        
        Args:
            field_value: String containing "entity, idxN" pairs
            idx_mapping: Dictionary mapping old idx to new idx
            
        Returns:
            Updated string with new idx values, invalid references removed
        """
        if not field_value or pd.isna(field_value):
            return field_value
            
        original_value = str(field_value)
        
        # Split by comma and process pairs
        items = [item.strip() for item in original_value.split(',')]
        updated_items = []
        changes_made = []
        removed_refs = []
        
        i = 0
        while i < len(items):
            if i + 1 < len(items) and items[i + 1].startswith('idx'):
                # This is an "entity, idxN" pair
                entity = items[i]
                idx_str = items[i + 1]
                
                # Extract old idx number
                try:
                    old_idx = int(idx_str.replace('idx', ''))
                    # Map to new idx if remapped
                    new_idx = idx_mapping.get(old_idx, old_idx)
                    if old_idx != new_idx:
                        changes_made.append(f"idx{old_idx}→idx{new_idx}")

                    # Validate entity-index consistency if context provided
                    if idx_to_ent is not None and ent_to_primary_idx is not None:
                        mapped_entity = idx_to_ent.get(new_idx)
                        if mapped_entity is None:
                            # idx doesn't exist in current study; drop the pair
                            removed_refs.append(f"{entity}, idx{new_idx}")
                        elif mapped_entity != entity:
                            # If entity mismatch, rewrite to the entity's primary idx if available
                            primary_idx = ent_to_primary_idx.get(entity)
                            if primary_idx is not None:
                                updated_items.extend([entity, f'idx{int(primary_idx)}'])
                            else:
                                # No valid idx for this entity; drop the pair
                                removed_refs.append(f"{entity}, idx{new_idx}")
                        else:
                            updated_items.extend([entity, f'idx{new_idx}'])
                    else:
                        # No validation context; just apply remap
                        updated_items.extend([entity, f'idx{new_idx}'])
                except (ValueError, AttributeError):
                    # If parsing fails, keep original
                    updated_items.extend([entity, idx_str])
                
                i += 2  # Skip both entity and idx
            else:
                # Single item, keep as is
                updated_items.append(items[i])
                i += 1
        
        result = ', '.join(updated_items)
        
        # Debug output for significant changes (only if verbose debugging is enabled)
        # This will be controlled by the calling function
        if changes_made and len(changes_made) > 0:
            # Store debug info for later display if needed
            pass
        
        if removed_refs and len(removed_refs) > 0:
            # Store debug info for removed references
            pass
        
        return result
    
    def _clean_invalid_references(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove invalid references from evidence and associate fields.
        A reference is invalid if the referenced entity and idx combination doesn't exist in the same study.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            DataFrame with invalid references removed
        """
        print("\\n=== CLEANING INVALID REFERENCES ===")
        
        cleaned_df = df.copy()
        total_cleaned = 0
        
        for study_id, group in cleaned_df.groupby('study_id'):
            study_data = cleaned_df[cleaned_df['study_id'] == study_id]
            
            # Create a set of valid (ent, idx) pairs for this study
            valid_pairs = set()
            for _, row in study_data.iterrows():
                valid_pairs.add((row['ent'], row['idx']))
            
            # Clean evidence and associate fields
            for idx, row in study_data.iterrows():
                for field in ['evidence', 'associate']:
                    if pd.notna(row[field]) and str(row[field]).strip():
                        original_value = str(row[field])
                        cleaned_value = self._clean_field_references(original_value, valid_pairs)
                        
                        if original_value != cleaned_value:
                            cleaned_df.loc[idx, field] = cleaned_value
                            total_cleaned += 1
        
        print(f"Cleaned {total_cleaned} invalid references")
        print("=== END CLEANING INVALID REFERENCES ===\\n")
        
        return cleaned_df
    
    def _clean_field_references(self, field_value: str, valid_pairs: set) -> str:
        """
        Clean a single field by removing invalid references.
        
        Args:
            field_value: String containing "entity, idxN" pairs
            valid_pairs: Set of valid (entity, idx) tuples
            
        Returns:
            Cleaned string with invalid references removed
        """
        if not field_value or pd.isna(field_value):
            return field_value
        
        # Split by comma and process pairs
        items = [item.strip() for item in field_value.split(',')]
        cleaned_items = []
        
        i = 0
        while i < len(items):
            if i + 1 < len(items) and items[i + 1].startswith('idx'):
                # This is an "entity, idxN" pair
                entity = items[i]
                idx_str = items[i + 1]
                
                # Extract idx number
                try:
                    idx_num = int(idx_str.replace('idx', ''))
                    
                    # Check if this (entity, idx) pair is valid
                    if (entity, idx_num) in valid_pairs:
                        cleaned_items.extend([entity, idx_str])
                    # If invalid, skip this pair (remove it)
                    
                except (ValueError, AttributeError):
                    # If parsing fails, keep original
                    cleaned_items.extend([entity, idx_str])
                
                i += 2  # Skip both entity and idx
            else:
                # Single item, keep as is
                cleaned_items.append(items[i])
                i += 1
        
        return ', '.join(cleaned_items)
    
    def process(self):
        """Process the data and return both section and report level dataframes."""
        print("\n=== PROCESSING SECTION LEVEL ===")
        section_level_df = self.create_section_level_df()
        
        print("\n=== PROCESSING REPORT LEVEL ===")
        report_level_df = self.create_report_level_df()
        
        return section_level_df, report_level_df
    
class PathwayProcessor:
    """Processes medical entities and view information to determine appropriate pathways."""
    
    def __init__(self, grouped_dict: Dict[str, Any]):
        """
        Initialize the PathwayProcessor.
        
        Args:
            grouped_dict: Dictionary containing pathway information for different entities
        """
        global global_vocab_processor
        self.grouped_dict = grouped_dict
        self.search_columns = [
            'morphology', 'distribution', 'measurement', 'severity', 'comparison', 'onset', 
            'no change', 'improved', 'worsened', 'placement', 'past hx', 'other source', 
            'assessment limitations'
        ]
        
        # Entities that should generate error messages when no pathway is found
        # Note: These should be the normalized entity names that exist in grouped_dict
        self.critical_entities = {
            'pneumonia', 'atelectasis', 'copd', 'bronchitis', 'congestive heart failure', 
            'pulmonary edema', 'pneumothorax', 'emphysema', 'consolidation', 'chf', 
            'lung cancer', 'pleural effusion', 'cardiomegaly', 'tension pneumothorax', 
            'effusion', 'malignancy', 'lung malignancy', 'lung carcinoma', 'carcinoma', 
            'edema', 'fracture', 'pneumonitis', 'chronic obstructive pulmonary disease', 
            'tuberculosis'
        }
        
        # Entity-specific search terms mapping
        self.entity_search_terms = {
            'pleural effusion': ['loculated'],
            'fracture': ['acute', 'healed', 'subacute to chronic'],
            'emphysema': ['severe'],
            'tuberculosis': ['active', 'non-active', 'chronic'],
            'bronchitis': ['acute', 'chronic']
        }

    
    def get_specific_term_from_row(self, row: pd.Series, ent_value: str, key: str) -> str:
        """
        Find specific terms in the row that modify the entity.
        
        Args:
            row: DataFrame row containing entity attributes
            ent_value: Original entity value
            key: Entity key to look up search terms
            
        Returns:
            Modified entity string if specific terms are found, otherwise original entity
        """
        # Get search terms for this entity
        search_terms = self.entity_search_terms.get(key, [])
        if not search_terms:
            return ent_value
        
        # Find matching terms in the row
        found_terms = []
        for col in self.search_columns:
            if pd.notna(row[col]) and isinstance(row[col], str):
                for term in search_terms:
                    if term in row[col].lower() and term not in found_terms:
                        found_terms.append(term)
        
        # Combine found terms with the entity
        if found_terms:
            combined_terms = ''.join(found_terms)
            return f"{combined_terms} {key}"
            
        return ent_value
    
    def extract_loc1_value(self, item: str) -> Optional[str]:
        """
        Extract the first location value from a location string.
        
        Args:
            item: Location string to parse
            
        Returns:
            Extracted location value or None if not found
        """
        if not isinstance(item, str) or not item.strip():
            return None
            
        # Simply return the first part of the location string
        # Split by comma and return the first non-empty part
        parts = [part.strip() for part in item.split(',') if part.strip()]
        return parts[0] if parts else None
    
    def check_location_category(self, obs_cat: Union[str, List[str]], 
                               obs_cat2: Union[str, List[str]], 
                               category: str, 
                               subcategory: Optional[str] = None) -> bool:
        """
        Check if the observation categories match the specified category and subcategory.
        
        Args:
            obs_cat: Primary observation category
            obs_cat2: Secondary observation category
            category: Category to check for
            subcategory: Subcategory to check for (optional)
            
        Returns:
            True if categories match, False otherwise
        """
        if isinstance(obs_cat, list) and isinstance(obs_cat2, list):
            for cat, subcat in zip(obs_cat, obs_cat2):
                if cat == category:
                    if subcategory is None:
                        return True
                    if isinstance(subcat, str) and subcategory in subcat:
                        return True
            return False
        
        if obs_cat != category:
            return False
            
        if subcategory is None:
            return True
            
        return isinstance(obs_cat2, str) and subcategory in obs_cat2
    
    def check_spine(self, obs_cat: Union[str, List[str]], obs_cat2: Union[str, List[str]]) -> bool:
        """Check if location is in spine category."""
        return self.check_location_category(obs_cat, obs_cat2, 'location', 'Musculoskeletal > Bones > Spine')
    
    def check_bone(self, obs_cat: Union[str, List[str]], obs_cat2: Union[str, List[str]]) -> bool:
        """Check if location is in bone category but not spine."""
        if self.check_location_category(obs_cat, obs_cat2, 'location', 'Musculoskeletal > Bones'):
            return not self.check_spine(obs_cat, obs_cat2)
        return False
    
    def calculate_jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """
        Calculate Jaccard similarity between two sets.
        
        Args:
            set1: First set of strings
            set2: Second set of strings
            
        Returns:
            Jaccard similarity coefficient (0-1)
        """
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0
    
    def _normalize_view_token(self, token: str) -> str:
       """Normalize a single view token to canonical form (AP/PA/LATERAL/LL)."""
       if not isinstance(token, str):
           return token
       t = token.strip().lower()
       if t in {"ap", "antero posterior", "antero-posterior"}:
           return "AP"
       if t in {"pa", "postero anterior", "postero-anterior"}:
           return "PA"
       if t in {"lat", "lateral"}:
           return "LATERAL"
       if t in {"ll", "left lateral"}:
           return "LL"
       # posture tokens kept as-is to allow later filtering
       if t in {"erect", "recumbent", "supine", "semi-erect", "semi erect"}:
           return t
       return token.strip()

    def preprocess_view_info(self, view_info):
       """Convert view_info string to list of view tokens with proper cleaning.
       - Clean list-like strings and NaN values
       - Handle malformed view_info strings
       """
       if pd.isna(view_info):
           return []
       
       # Clean the string first
       cleaned = str(view_info).strip().lower()
       
       # Remove list representations and clean up
       cleaned = cleaned.replace('[', '').replace(']', '').replace("'", "")
       cleaned = cleaned.replace('nan', '').replace('none', '')
       
       # Split by comma and clean each token
       view_info_list = []
       for token in cleaned.split(','):
           token = token.strip()
           if token and token not in ['', 'nan', 'none']:
               view_info_list.append(token)
       
       return view_info_list
    
    def normalize_entity_name(self, ent_value: str) -> str:
        """Normalize entity names to match grouped_dict keys.
        
        Args:
            ent_value: The entity value to normalize
            
        Returns:
            Normalized entity name
        """
        if pd.isna(ent_value):
            return ent_value
            
        ent_value = ent_value.strip().lower()
        
        # Handle common entity name variations
        entity_mappings = {
            'effusion': 'pleural effusion',
            'lung malignancy': 'lung cancer',
            'malignancy': 'lung cancer',
            'pneumothoraces': 'pneumothorax',
            'fractures': 'fracture',
            'fx': 'fracture',
            'chf': 'congestive heart failure',
            'copd': 'chronic obstructive pulmonary disease',
            'pna': 'pneumonia',
            'pneumonitis': 'pneumonia',
            'active tb': 'tuberculosis',  # Map to 'tuberculosis' since 'active tuberculosis' queries only contain 'tb' and 'tuberculosis'
            'active tuberculosis': 'tuberculosis',  # Map to 'tuberculosis' since queries don't include 'active tuberculosis'
            'tb': 'tuberculosis',
            'ptx': 'pneumothorax',
            'pneumo': 'pneumonia',
            'pulm edema': 'pulmonary edema',
            'atelectases': 'atelectasis',
            'consolidations': 'consolidation',
            'pneumonias': 'pneumonia',
            'lung carcinoma': 'lung cancer',
            'carcinoma': 'lung cancer'
        }
        
        return entity_mappings.get(ent_value, ent_value)
   
    def get_pathway_for_ent_and_view(self, ent_value: str, view_info: str, row: pd.Series) -> Optional[List[str]]:
        """
        Determine the appropriate pathway based on entity value and view information.
        
        Args:
            ent_value: Entity value
            view_info: View information string
            row: DataFrame row containing entity attributes
            
        Returns:
            List of pathway strings or None if no pathway is found
        """
        if pd.isna(ent_value) or pd.isna(view_info):
            return None
        
        # Normalize strings
        ent_value = self.normalize_entity_name(ent_value)
        view_info = view_info.strip().lower()
        # view_info_list = [v.strip() for v in view_info.split(',')]
        view_info_list = self.preprocess_view_info(view_info) # clean up view info list -> might be cleaner if we do this inside the ViewInformationProcessor
                
        # Handle fracture entities specially
        if ent_value in ["fractures", "fx", "fracture"]:
            location = row['location']
            if pd.notna(location):
                # loc1_value = self.extract_loc1_value(location)
                
                # if loc1_value:
                obs_cat, obs_cat2 = global_vocab_processor.find_cat(location)  # Assuming find_cat is defined elsewhere
                if obs_cat == [] and obs_cat2 == []:
                    return None
                if self.check_spine(obs_cat, obs_cat2): # check if location of fracture is in the spine
                    return self.grouped_dict['spinal fracture'].get('pathway', [])
                elif self.check_bone(obs_cat, obs_cat2): # check if fracture is in a particular bone
                    return self.grouped_dict['fracture'].get('pathway', [])
                # in all other cases, the fracture pathway does not apply, so don't use the fracture pathway
                return None
        
        # Track best pathway match
        max_overlap = 0
        best_pathways = []
        
        if self.grouped_dict is None:
            print("Warning: grouped_dict is None")
            return None

        # Search through all entities in grouped_dict -> search the pathways for a match
        for key, value in self.grouped_dict.items():
            if not isinstance(value, dict) or 'queries' not in value:
                continue
                
            # Check if entity matches any of the queries
            queries = [syn.strip().lower() for syn in value['queries']]
            if ent_value not in queries:
                continue
            
            # Check for specific terms that modify the entity
            # e.g. loculated pleural effusion is a pathway, but entity would just be pleural effusion
            # so in that case, you have to check if "loculated" is part of any of the other relations (morphology, distribution, etc)
            # if it is, then we select the "loculated pleural effusion" pathway (first "if" statement after this)
            specific_term = self.get_specific_term_from_row(row, ent_value, key)
            
            if specific_term != ent_value:
                # If we found a specific term, use its pathway directly
                best_pathways = self.grouped_dict[specific_term].get('pathway', [])
                max_overlap = 1
            else:
                # Otherwise, find the best matching pathway based on view information
                # there are multiple pleural effusion pathways, each with a different list of views -> select the right one based on view
                pathways = value.get('pathway', [])
                for pathway in pathways:
                    if not isinstance(pathway, str):
                        continue
                        
                    # Extract view information from pathway
                    parts = pathway.split('>')
                    view_parts = [part.strip().lower() for part in parts if part.strip().lower().startswith('view:')]
                    
                    for view_part in view_parts:
                        pathway_view_content = view_part[len('view:'):].strip()
                        pathway_view_content_list = [v.strip() for v in pathway_view_content.split(',') if v.strip()]
                        
                        # Calculate Jaccard similarity between view info lists
                        # there might be multiple matches if we match on the condition that any of the views should be present in the pathway
                        # so we choose the pathway that has the closest total overlap in terms of view lists
                        overlap = self.calculate_jaccard_similarity(
                            set(view_info_list), 
                            set(pathway_view_content_list)
                        )
                        
                        if overlap > max_overlap:
                            max_overlap = overlap
                            best_pathways = [pathway]
        
        # Handle case where no pathway was found
        if max_overlap == 0:
            # Check if entity exists in grouped_dict with different names
            entity_found = False
            matching_keys = []
            for key, value in self.grouped_dict.items():
                if isinstance(value, dict) and 'queries' in value:
                    queries = [syn.strip().lower() for syn in value['queries']]
                    if ent_value in queries:
                        entity_found = True
                        matching_keys.append(key)
            
            if entity_found:
                # Entity exists but no view match - this is the real problem
                if ent_value in self.critical_entities:
                    print(f"ERROR: No pathway found (overlap: {max_overlap}) for entity '{ent_value}' with view_info '{view_info}'")
                    print(f"  Processed view_info_list: {view_info_list}")
                    print(f"  Found entity in keys: {matching_keys}")
                    print(f"  Available pathways for this entity: {[self.grouped_dict[key].get('pathway', []) for key in matching_keys]}")
                    input("STOP!!!")
            else:
                # Entity doesn't exist in grouped_dict at all
                if ent_value in self.critical_entities:
                    print(f"ERROR: Entity '{ent_value}' not found in grouped_dict")
                    print(f"  Available entities: {[key for key, value in self.grouped_dict.items() if isinstance(value, dict) and 'queries' in value]}")
                    input("STOP!!!")
            return None
            
        return best_pathways
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process a dataframe to add pathway information.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with added 'pathway' column
        """
        result_df = df.copy()
        result_df['pathway'] = result_df.apply(
            lambda x: self.get_pathway_for_ent_and_view(x['ent'], x['view_information'], x),
            axis=1
        )
        return result_df


class TextSimilarityUtils:
    """Utility class for text similarity calculations."""
    
    @staticmethod
    def jaccard_similarity(str1: Optional[str], str2: Optional[str]) -> float:
        """Calculate Jaccard similarity between two strings (word-level)."""
        if str1 is None or str2 is None:
            return 0.0
        if str1 is None and str2 is None:
            return 1.0

        set1, set2 = set(str1.lower().split()), set(str2.lower().split())
        return len(set1.intersection(set2)) / len(set1.union(set2)) if set1.union(set2) else 0

    @staticmethod
    def jaccard_similarity_char(str1: str, str2: str) -> float:
        """Calculate Jaccard similarity between two strings (character-level)."""
        set1, set2 = set(str1.lower()), set(str2.lower())
        return len(set1.intersection(set2)) / len(set1.union(set2)) if set1.union(set2) else 0
    
    @staticmethod
    def match_strings(obs_value: str, ent_value: str) -> bool:
        """Match strings using edit distance and Jaccard similarity."""
        global global_vocab_processor
        
        obs_queries = global_vocab_processor.find_queries(obs_value) # find all queries in vocabulary that match the observation value
        
        for obs_syn in obs_queries:
            edit_distance = Levenshtein.distance(obs_syn, ent_value) # calculate string distance between observation name and entity name we want to compare with
            if obs_syn == ent_value:
                return True
            else:
                jaccard_score = TextSimilarityUtils.jaccard_similarity(obs_syn, ent_value) # calculate jaccard similarity on word level
                if jaccard_score >= 0.7:
                    return True
                elif edit_distance <= 5:
                    char_jaccard_score = TextSimilarityUtils.jaccard_similarity_char(obs_syn, ent_value) # calculate jaccard similarity on character level
                    if char_jaccard_score >= 0.5:
                        return True
            return False
        return False


class StatusAnalyzer:
    """Handles status normalization and conflict resolution for medical reports."""
    
    def __init__(self, verbose: bool = False, matching: str = 'model'):
        self.positive_statuses = {'dp', 'tp'}
        self.negative_statuses = {'dn', 'tn'}
        self.subgroup_categories = [
            'ent', 'study_id', 'location', 'placement', 'other source', 
            'assessment limitations', 'past hx', 'morphology', 'distribution', 
            'measurement', 'severity', 'comparison', 'onset', 'improved', 
            'no change', 'worsened'
        ]
        self.merge_cols = [
            'ent', 'study_id', 'evidence', 'associate', 'location', 'placement',
            'other source', 'assessment limitations', 'past hx', 'morphology', 
            'distribution', 'measurement', 'severity', 'comparison', 'onset',
            'improved', 'no change', 'worsened'
        ]
        self.verbose = verbose
        self.matching = matching
        # thresholds
        self.model_threshold = 0.95
        self.string_threshold = 0.7
        # lazy model holder
        self._embedder = None
        # merge statistics tracker (required by process_subgroups)
        self.merge_stats = {
            'total_subgroups': 0,
            'merged_subgroups': 0,
            'total_rows_before': 0,
            'total_rows_after': 0,
            'merged_details': []
        }
        # blacklist categories (pairs of mutually exclusive tokens)
        self._blacklist_pairs = {
            # directions / sides
            ('left', 'right'), ('left-sided', 'right-sided'), ('left-sided', 'right'), ('left', 'right-sided'),
            ('upper', 'lower'), ('mid', 'lower'), ('upper', 'mid'), ('upper', 'middle'), ('middle', 'lower'),
            ('anterior', 'posterior'), ('anterior', 'lateral'), ('posterior', 'lateral'),
            ('superior', 'inferior'), ('apical', 'basal'), ('central', 'peripheral'), ('proximal', 'distal'),
            ('medial', 'lateral'), ('ventral', 'dorsal'),
            # cardio/pulmo examples
            ('cardiac', 'pulmonary'), ('mediastinal', 'pleural'), ('pericardial', 'pleural'),
            # disease pairs
            ('effusion', 'pneumothorax'), ('effusion', 'edema'), ('effusion', 'atelectasis'), ('effusion', 'consolidation'), ('effusion', 'pneumonia'),
            ('consolidation', 'opacification'), ('consolidation', 'atelectasis'), ('consolidation', 'edema'),
            ('atelectasis', 'pneumonia'), ('atelectasis', 'aeration'), ('atelectasis', 'opacity'),
            ('pneumonia', 'edema'),
            # mass/nodule etc
            ('mass', 'consolidation'), ('mass', 'opacification'), ('mass', 'opacity'), ('mass', 'atelectasis'), ('nodule', 'consolidation'),
        }

    def _contains_token(self, text: str, token: str) -> bool:
        return token in set(str(text).lower().split())

    def _apply_blacklist_penalty(self, s1: str, s2: str, sim: float) -> float:
        """Reduce similarity if blacklist conflicting tokens appear across texts."""
        if sim <= 0:
            return sim
        w1 = set(str(s1).lower().split())
        w2 = set(str(s2).lower().split())
        # quick reject if texts are too short
        if not w1 or not w2:
            return sim
        for a, b in self._blacklist_pairs:
            if (a in w1 and b in w2) or (a in w2 and b in w1):
                # strong penalty: treat as mismatch
                penalized = min(sim, 0.2)
                if self.verbose:
                    print(f"Blacklist penalty applied: ({a},{b}) sim {sim:.3f} -> {penalized:.3f}")
                return penalized
        return sim

    # ============ Matching helpers ============
    def _ensure_model(self):
        if self._embedder is not None:
            return
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            class _Embedder:
                def __init__(self):
                    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    self.tokenizer = AutoTokenizer.from_pretrained('FremyCompany/BioLORD-2023', model_max_length=512, clean_up_tokenization_spaces=True)
                    self.model = AutoModel.from_pretrained('FremyCompany/BioLORD-2023').to(self.device)
                def encode(self, texts: list) -> np.ndarray:
                    import torch.nn.functional as F
                    import torch
                    enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
                    with torch.no_grad():
                        out = self.model(**enc)
                    # mean pooling
                    token_embeddings = out[0]
                    input_mask_expanded = enc['attention_mask'].unsqueeze(-1).expand(token_embeddings.size()).float()
                    pooled = (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)
                    emb = F.normalize(pooled, p=2, dim=1)
                    return emb.detach().cpu().numpy()
                def cosine(self, a: list, b: list) -> float:
                    import numpy as _np
                    ea = self.encode([a])[0]
                    eb = self.encode([b])[0]
                    return float((_np.dot(ea, eb) / (np.linalg.norm(ea) * np.linalg.norm(eb) + 1e-9)))
            self._embedder = _Embedder()
        except Exception as e:
            if self.verbose:
                print(f"Model load failed, fallback to string similarity. Reason: {e}")
                raise e
            self.matching = 'string'

    def _similar(self, s1: str, s2: str) -> float:
        if not isinstance(s1, str) or not isinstance(s2, str):
            return 0.0
        s1c, s2c = s1.strip().lower(), s2.strip().lower()
        if s1c == '' or s2c == '':
            return 0.0
        if self.matching == 'model':
            self._ensure_model()
            try:
                base = self._embedder.cosine(s1c, s2c)
                return self._apply_blacklist_penalty(s1c, s2c, base)
            except Exception:
                # fallback to string
                pass
        # string-based jaccard + char jaccard + Levenshtein heuristic
        words1, words2 = set(s1c.split()), set(s2c.split())
        inter = len(words1 & words2)
        union = len(words1 | words2) or 1
        jacc = inter / union
        if jacc >= self.string_threshold:
            return self._apply_blacklist_penalty(s1c, s2c, jacc)
        # char-level
        c1, c2 = set(s1c), set(s2c)
        cj = len(c1 & c2) / (len(c1 | c2) or 1)
        if cj >= 0.5:
            return self._apply_blacklist_penalty(s1c, s2c, max(jacc, cj))
        # edit distance scaled
        dist = Levenshtein.distance(s1c, s2c)
        scale = max(len(s1c), len(s2c)) or 1
        sim = 1.0 - (dist / scale)
        return self._apply_blacklist_penalty(s1c, s2c, max(jacc, cj, sim))

    def _consolidate_column(self, series: pd.Series, threshold: float) -> pd.Series:
        vals = [v for v in series.astype(str).tolist()]
        if not vals:
            return series
        reps = {}
        for i, v in enumerate(vals):
            if v in reps:
                continue
            reps[v] = v  # self as representative
            for j in range(i+1, len(vals)):
                u = vals[j]
                if u in reps:
                    continue
                sim = self._similar(v, u)
                if sim >= threshold:
                    reps[u] = v
        # map
        mapped = series.astype(str).map(lambda x: reps.get(x, x))
        return mapped

    def _apply_similarity_consolidation(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        df = df.copy()
        thr = self.model_threshold if self.matching == 'model' else self.string_threshold
        # Build composite phrases: ent + location + non-empty VALID_ATTR_CATEGORY columns
        def _row_phrase(row: pd.Series) -> str:
            parts = []
            ent = str(row.get('ent', '') or '').strip()
            loc = str(row.get('location', '') or '').strip()
            if ent:
                parts.append(ent)
            if loc:
                parts.append(loc)
            for col in VALID_ATTR_CATEGORY:
                if col in row.index:
                    val = str(row[col]) if pd.notna(row[col]) else ''
                    val = val.strip()
                    if val and val.lower() != 'nan':
                        parts.append(val)
            # join with single spaces to make a natural phrase
            return ' '.join(parts).lower()

        phrases = df.apply(_row_phrase, axis=1)
        indices = df.index.tolist()
        rep_for = {idx: idx for idx in indices}
        for i in range(len(indices)):
            idx_i = indices[i]
            pi = phrases.iloc[i]
            for j in range(i+1, len(indices)):
                idx_j = indices[j]
                if rep_for[idx_j] != idx_j:
                    continue
                pj = phrases.iloc[j]
                # string 모드에서는 ent/loc 개별 유사도의 곱을 기준으로 판단
                if self.matching == 'string':
                    ent_i = str(df.at[idx_i, 'ent']) if 'ent' in df.columns and pd.notna(df.at[idx_i, 'ent']) else ''
                    ent_j = str(df.at[idx_j, 'ent']) if 'ent' in df.columns and pd.notna(df.at[idx_j, 'ent']) else ''
                    loc_i = str(df.at[idx_i, 'location']) if 'location' in df.columns and pd.notna(df.at[idx_i, 'location']) else ''
                    loc_j = str(df.at[idx_j, 'location']) if 'location' in df.columns and pd.notna(df.at[idx_j, 'location']) else ''
                    s_ent = self._similar(ent_i, ent_j) if ent_i or ent_j else 1.0
                    s_loc = self._similar(loc_i, loc_j) if loc_i or loc_j else 1.0
                    sim = s_ent * s_loc
                else:
                    sim = self._similar(pi, pj)
                if sim >= thr:
                    # map j to i's representative
                    rep_for[idx_j] = rep_for[idx_i]
        # apply representatives: set ent/location to representative's values
        for idx in indices:
            ridx = rep_for[idx]
            if ridx != idx:
                df.at[idx, 'ent'] = df.at[ridx, 'ent']
                if 'location' in df.columns:
                    df.at[idx, 'location'] = df.at[ridx, 'location']
        merged = sum(1 for k,v in rep_for.items() if k!=v)
        if self.verbose and merged > 0:
            print(f"mode={self.matching} threshold={thr}")
            print("sample phrases:")
            for s in phrases.head(5).tolist():
                print("  -", s)
            print(f"merged pairs: {merged}")
        return df
    
    def normalize_statuses(self, subgroup):
        """Normalize and resolve conflicts in status values."""
        statuses = [s.split('|')[0] if '|' in s else s for s in subgroup['status'].unique()]
        unique_statuses = set(statuses)
        
        if len(unique_statuses) <= 1: # if all statuses are the same, we can just use that one
            return subgroup.assign(status=statuses[0])
        
        # Handle status conflicts
        has_positive = any(s in self.positive_statuses for s in statuses)
        has_negative = any(s in self.negative_statuses for s in statuses)
        
        if has_positive and has_negative: # positive and negative conflict
            # Only log for specific study to avoid spam
            study_id = subgroup['study_id'].iloc[0] if 'study_id' in subgroup.columns else 'unknown'
            # if study_id == 's59542064':
            #     print(f"\n🚨 P vs N conflict detected in s59542064, removing subgroup:")
            #     print(f"Entities: {subgroup['ent'].unique()}")
            #     print(f"Statuses: {subgroup['status'].unique()}")
            #     print(f"Locations: {subgroup['location'].unique()}")
            #     input("Press Enter to continue after P vs N conflict detection...")
            return None
        
        # we will never enter this line normally
        new_status = 'dp' if all(s in self.positive_statuses for s in statuses) else 'dn'
        return subgroup.assign(status=new_status)
    
    def merge_duplicate_rows(self, subgroup):
        """Merge multiple rows in a subgroup into a single row."""
        if len(subgroup) < 2:
            return subgroup
        
        # Track merge statistics
        self.merge_stats['merged_subgroups'] += 1
        self.merge_stats['total_rows_before'] += len(subgroup)
        self.merge_stats['total_rows_after'] += 1
        
        # Log significant merges for specific study
        study_id = subgroup['study_id'].iloc[0] if 'study_id' in subgroup.columns else 'unknown'
        if study_id == 's59542064' and len(subgroup) > 2:
            print(f"\n📊 Merging {len(subgroup)} rows in s59542064:")
            print(f"Entity: {subgroup['ent'].iloc[0]}")
            print(f"Statuses: {subgroup['status'].unique()}")
        
        combined_row = subgroup.iloc[0].copy()
        merge_details = {
            'study_id': subgroup['study_id'].iloc[0],
            'ent': subgroup['ent'].iloc[0],
            'row_count': len(subgroup),
            'merged_columns': {}
        }
        
        # extra columns which were not used for grouping: evidence, associate
        # these need to be merged if they differ for the duplicate rows -> just join them by ',' 
        for col in self.merge_cols:
            values = subgroup[col].astype(str).str.lower().unique()
            filtered = [v.strip() for v in values if v.strip() and v.lower() != 'nan']
            
            # Track columns that had multiple values
            if len(filtered) > 1:
                merge_details['merged_columns'][col] = filtered
                
            combined_row[col] = ', '.join(filtered) if filtered else ''
        
        if self.verbose and merge_details['merged_columns']:
            self.merge_stats['merged_details'].append(merge_details)
            print(f"\n🔄 Merged {len(subgroup)} rows for entity '{merge_details['ent']}' (study_id: {merge_details['study_id']})")
            print(f"   Merged columns: {list(merge_details['merged_columns'].keys())}")
            for col, values in merge_details['merged_columns'].items():
                print(f"   - {col}: {values} → '{combined_row[col]}'")
        
        return pd.DataFrame([combined_row])
    
    def process_subgroups(self, group_rows):
        """Process subgroups by normalizing statuses and merging duplicates."""
        if group_rows.empty:
            return pd.DataFrame()
        
        result_list = []
        group_rows = group_rows.fillna('')
        # Similarity-based consolidation before grouping (ent/location)
        group_rows = self._apply_similarity_consolidation(group_rows)
        
        # Group rows by the specified categories
        subgroups = group_rows.groupby(self.subgroup_categories)
        self.merge_stats['total_subgroups'] += len(subgroups)
        
        for _, subgroup in subgroups:
            # Normalize statuses
            normalized_subgroup = self.normalize_statuses(subgroup)
            if normalized_subgroup is None:  # If conflict is detected (P and N in same group) the rows are removed
                continue
            
            # Merge duplicate rows
            processed_subgroup = self.merge_duplicate_rows(normalized_subgroup)
            result_list.append(processed_subgroup)
        
        return pd.concat(result_list, ignore_index=True) if result_list else pd.DataFrame()
    
    def print_merge_summary(self):
        """Print a summary of the merge operations."""
        print("\n===== MERGE SUMMARY =====")
        print(f"Total subgroups processed: {self.merge_stats['total_subgroups']}")
        print(f"Subgroups with merges: {self.merge_stats['merged_subgroups']} ({self.merge_stats['merged_subgroups']/max(1, self.merge_stats['total_subgroups'])*100:.1f}%)")
        print(f"Total rows before merging: {self.merge_stats['total_rows_before']}")
        print(f"Total rows after merging: {self.merge_stats['total_rows_after']}")
        print(f"Rows reduced: {self.merge_stats['total_rows_before'] - self.merge_stats['total_rows_after']} ({(self.merge_stats['total_rows_before'] - self.merge_stats['total_rows_after'])/max(1, self.merge_stats['total_rows_before'])*100:.1f}%)")
        
        if self.verbose:
            print("\nTop merged columns:")
            column_counts = {}
            for detail in self.merge_stats['merged_details']:
                for col in detail['merged_columns']:
                    column_counts[col] = column_counts.get(col, 0) + 1
            
            for col, count in sorted(column_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"- {col}: {count} merges")


class StatusProcessor:
    """Handles status transitions in medical pathways."""
    
    def __init__(self):
        # key: status of diagnosis as written in the report
        # value: dict indicating how to change each status of observations in the pathway
        self.status_map = {
            'dp': {'dp': 'dp', 'dn': 'dn', 'tp': 'tp', 'tn': 'tn'}, # if definitive, everything remains as it is
            'tp': {'dp': 'tp', 'dn': 'tn', 'tp': 'tp', 'tn': 'tn'}, # if tentative, everything becomes tentative
            'dn': {'dp': None, 'dn': None, 'tp': None, 'tn': None}, # if negative, we don't know if pathway can be reversed, so don't apply
            'tn': {'dp': None, 'dn': None, 'tp': None, 'tn': None}  # idem
        }
    
    def switch_status(self, current_status: str, match: re.Match) -> str:
        """Apply status transition rules."""
        status = match.group(1)
        
        # status of observation depends on status of diagnosis in report
        if current_status in self.status_map and status in self.status_map[current_status]:
            return f'status: {self.status_map[current_status][status]}'
        
        return f'status: {status}'


class PathwayExtractor:
    """Extracts and processes observations from pathways."""
    
    @staticmethod
    def extract_obs(modified_path: Union[str, List[str]]) -> List[str]:
        """Extract observation values from a pathway."""
        if isinstance(modified_path, list):
            all_obs = []
            for path in modified_path:
                obs_in_path = re.findall(r'obs: ([\w\s]+) \(', path)
                all_obs.extend(obs_in_path)
            return all_obs
        elif isinstance(modified_path, str):
            return re.findall(r'obs: ([\w\s]+) \(', modified_path)
        else:
            raise ValueError("OBS parsing error: Input must be string or list")


class LocationProcessor:
    """Processes location information in pathways."""
    
    @staticmethod
    def process_loc_in_obs(modified_path: Union[str, List[str]], cur_loc_value: str) -> Union[str, List[str]]:
        """Process location information in observations."""
        # print("\n=== Location Processing Debug ===")
        # print(f"Input - modified_path: {modified_path}")
        # print(f"Input - cur_loc_value: {cur_loc_value}")
        
        # Handle list input
        if isinstance(modified_path, str) and modified_path.startswith('['):
            try:
                modified_path = eval(modified_path)
                # print(f"List conversion - converted to: {modified_path}")
            except Exception:
                # print("List conversion failed:", modified_path)
                input("Press Enter to continue after list conversion failed...")
                return modified_path
        
        if isinstance(modified_path, list):
            # print("Processing list of paths")
            return [LocationProcessor.process_loc_in_obs(single_path, cur_loc_value) for single_path in modified_path]
        
        # Process single path
        view_match = re.match(r'(view: [^>]+) > (.*)', modified_path)
        if not view_match:
            print("No view match found:", modified_path)
            return modified_path
        
        view_part = view_match.group(1).strip()
        rest_of_path = view_match.group(2).strip()
        # print(f"\nPath parsing:")
        # print(f"- View part: {view_part}")
        # print(f"- Rest of path: {rest_of_path}")
        
        segments = re.split(r'[&,]+', rest_of_path)
        processed_segments = [view_part]
        # print(f"\nSegments found: {segments}")
        
        for segment in segments:
            segment = segment.strip()
            # print(f"\nProcessing segment: {segment}")
            
            if 'obs:' in segment:
                # Process location in observation
                loc_match = re.search(r'loc: ([\w\s]+)', segment)
                
                if loc_match:
                    pathway_loc = loc_match.group(1)
                    # print(f"Location found in pathway: {pathway_loc}")
                    # print(f"Current location value: {cur_loc_value}")
                    
                    if pathway_loc != cur_loc_value:
                        # Simply append the current location value
                        updated_loc = f"loc: {cur_loc_value}, {pathway_loc}"
                    else:
                        updated_loc = f'loc: {pathway_loc}'
                    
                    segment = re.sub(r'loc: [\w\s]+', updated_loc, segment)
                    # print(f"Updated segment: {segment}")
                else:
                    # Add location if not present
                    match = re.search(r'\((\w+)', segment)
                    obs_type = match.group(1) if match else None
                    # print(f"Observation type: {obs_type}")
                    
                    no_loc_types = {'cof', 'ncd', 'patient', 'patient info.'}
                    if not obs_type or obs_type not in no_loc_types:
                        segment = segment + f' > loc: {cur_loc_value}'
                        # print(f"Added location to segment: {segment}")
            
            processed_segments.append(' && ' + segment)
        
        result = ''.join(processed_segments)
        # print(f"\nFinal processed path: {result}")
        # print("=== End Location Processing ===\n")
        return result
    
    @staticmethod
    def process_paths(modified_paths: Union[str, List[str]], cur_loc_value: str) -> Union[str, List[str]]:
        """Process location information in multiple pathways."""
        if isinstance(modified_paths, list):
            return [LocationProcessor.process_loc_in_obs(path, cur_loc_value) for path in modified_paths]
        else:
            return LocationProcessor.process_loc_in_obs(modified_paths, cur_loc_value)
    
    # @staticmethod
    # def extract_loc_info(cur_loc_value: str) -> str:
    #     """Extract and format location information."""
    #     # Extract location and detail patterns
    #     patterns = {
    #         'loc1': re.search(r'loc1: ([\w\s]+)', cur_loc_value),
    #         'det1': re.search(r'det1: ([\w\s]+)', cur_loc_value),
    #         'loc2': re.search(r'loc2: ([\w\s]+)', cur_loc_value),
    #         'det2': re.search(r'det2: ([\w\s]+)', cur_loc_value),
    #         'loc3': re.search(r'loc3: ([\w\s]+)', cur_loc_value),
    #         'det3': re.search(r'det3: ([\w\s]+)', cur_loc_value)
    #     }
        
    #     loc_values = []
        
    #     # Process loc1 with det1
    #     if patterns['loc1']:
    #         loc1_value = patterns['loc1'].group(1).strip()
    #         if patterns['det1']:
    #             loc1_value = f"{loc1_value} {patterns['det1'].group(1).strip()}"
    #         loc_values.append(loc1_value)
        
    #     # Process loc2 with det2
    #     if patterns['loc2']:
    #         loc2_value = patterns['loc2'].group(1).strip()
    #         if patterns['det2']:
    #             loc2_value = f"{loc2_value} {patterns['det2'].group(1).strip()}"
    #         loc_values.append(loc2_value)
        
    #     # Process loc3 with det3
    #     if patterns['loc3']:
    #         loc3_value = patterns['loc3'].group(1).strip()
    #         if patterns['det3']:
    #             loc3_value = f"{loc3_value} {patterns['det3'].group(1).strip()}"
    #         loc_values.append(loc3_value)
        
    #     return ', '.join(loc_values)
    
    @staticmethod
    def format_location(loc: str) -> str:
        """Format location string - now just returns the location as-is."""
        if pd.isna(loc):
            return loc
        
        return loc
    
    @staticmethod
    def preprocess_location(loc_str: Union[str, List[str]]) -> List[str]:
        """Preprocess location string for comparison."""
        # Handle list input
        if isinstance(loc_str, list):
            result = []
            for item in loc_str:
                if not pd.isna(item):
                    result.append(LocationProcessor.preprocess_location(item))
            return result
        
        # Handle NaN
        if pd.isna(loc_str):
            return []
        
        if not isinstance(loc_str, str):
            loc_str = str(loc_str)
        
        # Simply split by comma and return cleaned parts
        parts = [part.strip() for part in loc_str.split(',') if part.strip()]
        return parts
    
    @staticmethod
    def loc_jaccard_similarity(loc_list1: List[str], loc_list2: List[str]) -> float:
        """Calculate Jaccard similarity between location lists."""
        # Handle empty lists
        if not loc_list1 and not loc_list2:
            return 1.0
        if not loc_list1 or not loc_list2:
            return 0.0
        
        # Normalize lists
        def normalize_list(lst):
            try:
                if isinstance(lst, list):
                    if lst and isinstance(lst[0], list):
                        return [item[0] if isinstance(item, list) and item else None for item in lst]
                    return lst
                return [lst]
            except IndexError as e:
                print(f"IndexError in normalize_list with lst: {lst}")
                raise
        
        normalized_list1 = normalize_list(loc_list1)
        normalized_list2 = normalize_list(loc_list2)
        
        # Calculate similarities
        similarities = []
        for item1 in normalized_list1:
            for item2 in normalized_list2:
                similarities.append(TextSimilarityUtils.jaccard_similarity(item1, item2))
        
        return max(similarities) if similarities else 0.0
    
    @staticmethod
    def loc_model_similarity(loc_list1: List[str], loc_list2: List[str], 
                           status_analyzer: 'StatusAnalyzer' = None) -> float:
        """Calculate model-based similarity between location lists."""
        # Handle empty lists
        if not loc_list1 and not loc_list2:
            return 1.0
        if not loc_list1 or not loc_list2:
            return 0.0
        
        # Normalize lists
        def normalize_list(lst):
            try:
                if isinstance(lst, list):
                    if lst and isinstance(lst[0], list):
                        return [item[0] if isinstance(item, list) and item else None for item in lst]
                    return lst
                return [lst]
            except IndexError as e:
                print(f"IndexError in normalize_list with lst: {lst}")
                raise
        
        normalized_list1 = normalize_list(loc_list1)
        normalized_list2 = normalize_list(loc_list2)
        
        # Calculate similarities using model if available, otherwise fallback to Jaccard
        similarities = []
        for item1 in normalized_list1:
            for item2 in normalized_list2:
                if status_analyzer is not None:
                    # Use model-based similarity
                    similarities.append(status_analyzer._similar(item1, item2))
                else:
                    # Fallback to Jaccard similarity
                    similarities.append(TextSimilarityUtils.jaccard_similarity(item1, item2))
        
        return max(similarities) if similarities else 0.0


class PathwayAnalyzer:
    """Analyzes pathway structures and extracts components."""
    
    @staticmethod
    def extract_value_before_parenthesis(value: str) -> str:
        """Extract the part of a value before any parentheses."""
        match = re.match(r'([^()]+)', value)
        return match.group(1).strip() if match else value.strip()
    
    @staticmethod
    def analyze_pathway_with_index(pathway_list: List[str], entity: str) -> pd.DataFrame:
        """Analyze pathway structure and convert to DataFrame."""
        data = []
        pattern = re.compile(r'(obs|status|loc|attr): ([^>]+)')
        
        for path_idx, pathway in enumerate(pathway_list):
            # Process list of pathways
            if isinstance(pathway, list) or isinstance(pathway, np.ndarray):
                for individual_pathway in pathway:
                    PathwayAnalyzer._process_individual_pathway(
                        individual_pathway, path_idx, entity, pattern, data
                    )
            else:
                # Process single pathway
                PathwayAnalyzer._process_individual_pathway(
                    pathway, path_idx, entity, pattern, data
                )
        
        return pd.DataFrame(data)
    
    @staticmethod
    def _process_individual_pathway(pathway: str, path_idx: int, entity: str, 
                                   pattern: re.Pattern, data: List[Dict[str, Any]]) -> None:
        """Process an individual pathway and extract components."""
        rows = pathway.split(' && ')
        for row in rows:
            row_data = {'path_idx': path_idx, 'entity': entity}
            segments = row.split(' > ')
            for segment in segments:
                match = pattern.search(segment)
                if match:
                    key, value = match.groups()
                    if key == 'attr':
                        PathwayAnalyzer._process_attribute(value, row_data)
                    else:
                        PathwayAnalyzer._process_other_key(key, value, row_data)
            data.append(row_data)
    
    @staticmethod
    def _process_attribute(value: str, row_data: Dict[str, Any]) -> None:
        """Process attribute values in pathway."""
        try:
            # Extract base attribute value
            attr_value = PathwayAnalyzer.extract_value_before_parenthesis(value)
            
            # Split by '/' for multiple cases
            attr_parts = value.split('/')
            attr_parts = [itr.replace(attr_value, '').strip() if idx == 0 else itr 
                         for idx, itr in enumerate(attr_parts)]
            
            # Process each part
            for attr_idx, attr_part in enumerate(attr_parts):
                if attr_idx == 0:
                    match = re.search(r'\((\w+)', attr_part.strip())
                    main_type = match.group(1) if match else None
                else:
                    main_type = PathwayAnalyzer.extract_value_before_parenthesis(attr_part)
                
                # Add to row data
                if main_type in row_data:
                    row_data[main_type] += f", {attr_value}"
                else:
                    row_data[main_type] = attr_value
        except AttributeError:
            row_data['attr_issue'] = value.strip()
    
    @staticmethod
    def _process_other_key(key: str, value: str, row_data: Dict[str, Any]) -> None:
        """Process non-attribute keys in pathway."""
        if key == 'obs':
            # Extract observation category and subcategory
            # cat: allow spaces and dots (e.g., "patient info.")
            match = re.search(r'\(([^)]+)', value.strip())
            if match:
                first_seg = match.group(1).split('(')[0].strip().lower()
                obs_cat = ' '.join(first_seg.split())  # normalize inner spaces
            else:
                obs_cat = None

            # subcategory: inner parentheses after the first
            match = re.search(r'\([^()]+\s*\(([^)]+)\)', value.strip())
            if match:
                obs_subcat = ' '.join(match.group(1).strip().lower().split())
            else:
                obs_subcat = None

            row_data['cat'] = obs_cat
            row_data['subcategory'] = obs_subcat
        
        # Add extracted value to row data
        if key in row_data:
            row_data[key] += f", {PathwayAnalyzer.extract_value_before_parenthesis(value)}"
        else:
            row_data[key] = PathwayAnalyzer.extract_value_before_parenthesis(value)


class LocationComparator:
    """Compares locations between datasets."""
    
    @staticmethod
    def compare_locations(current_std: pd.DataFrame, modified_path_df: pd.DataFrame, 
                         ent_value: str, obs_itr: str, obs_idx: int, 
                         status_analyzer: 'StatusAnalyzer' = None) -> Tuple[float, List[str], List[str]]:
        """Compare locations between current study and pathway using model-based similarity."""

        # Get all locations for records where entity equals ent_value
        if 'location' in current_std.columns:
            current_possible_location = current_std.loc[current_std['ent'] == ent_value, 'location'].values.tolist()
            if len(current_possible_location) == 0:
                current_possible_location = None
        else:
            current_possible_location = None
        
        # Get location in pathway for observation we are currently trying to match (obs_itr, obs_idx)
        if 'location' in modified_path_df.columns:
            modified_location = modified_path_df.loc[
                (modified_path_df['ent'] == obs_itr) & 
                (modified_path_df.index == obs_idx), 'location'
            ].values
            
            if len(modified_location) > 0:
                modified_location = modified_location[0]
            else:
                modified_location = None
        else:
            modified_location = None
        
        # Preprocess locations
        processed_current_location = LocationProcessor.preprocess_location(current_possible_location)
        processed_modified_location = LocationProcessor.preprocess_location(modified_location)
        
        # Calculate similarity between locations using model-based similarity
        similarity = LocationProcessor.loc_model_similarity(
            processed_current_location, processed_modified_location, status_analyzer
        )
        
        return similarity, processed_current_location, processed_modified_location


class StudyProcessor:
    """Processes medical studies and applies pathways to entities."""
    
    def __init__(self, status_analyzer: 'StatusAnalyzer' = None):
        self.except_obs_list = ['pulmonary marking', 'volume loss']
        self.processed_pairs = []  # Store already processed matching_row.to_dict() values
        self.status_analyzer = status_analyzer
    
    def _match_strings(self, str1: str, str2: str) -> bool:
        """Check if two strings match (case-insensitive)."""
        if pd.isna(str1) or pd.isna(str2):
            return False
        return str1.strip().lower() == str2.strip().lower()
    
    def _match_entities_with_normed(self, obs_itr: str, ent_value: str) -> bool:
        """Check if pathway observation matches existing entity using both ent and normed_ent from vocab."""
        if pd.isna(obs_itr) or pd.isna(ent_value):
            return False
        
        # First check direct entity match
        if self._match_strings(obs_itr, ent_value):
            return True
        
        # Then check normalized entity match using global_vocab_processor
        global global_vocab_processor
        if global_vocab_processor is not None:
            # Get normed_term for the existing entity using vocab
            normed_terms = global_vocab_processor.vocab_df.loc[
                global_vocab_processor.vocab_df['target_term'] == ent_value.lower(), 'normed_term'
            ].dropna().unique()
            
            # Check if pathway observation matches any of the normed terms
            for normed_term in normed_terms:
                if not pd.isna(normed_term) and normed_term != '':
                    if self._match_strings(obs_itr, normed_term):
                        return True
        
        return False
    
    def _is_entity_related_to_disease(self, disease_row: pd.Series, ent_value: str, obs_itr: str, 
                                    current_std: pd.DataFrame, modified_path_df: pd.DataFrame, 
                                    obs_idx: int, report_type: str) -> bool:
        """
        Check if an entity is already related to the disease through evidence.
        This implements the core logic from the Jupyter notebook.
        """
        evidence_series = disease_row['evidence']
        
        if pd.isna(evidence_series) or evidence_series == '' or not isinstance(evidence_series, str):
            return False
        
        # Parse evidence items
        evidence_items = [item for item in evidence_series.split(', ') if not item.startswith('idx')]
        
        # Get matching sections
        if report_type == 'section-level':
            matching_sections = [disease_row['section']] if isinstance(disease_row['section'], str) else disease_row['section'].tolist()
            try:
                evidence_sections = [current_std[current_std['idx']==int(item.split('idx')[-1])]['section'].values[0] 
                                   for item in evidence_series.split(', ') if item.startswith('idx')]
            except (IndexError, ValueError):
                return False
        else:
            matching_sections = [disease_row['DxTP']] if isinstance(disease_row['DxTP'], str) else disease_row['DxTP'].tolist()
            try:
                evidence_sections = [current_std[current_std['idx']==int(item.split('idx')[-1])]['DxTP'].values[0] 
                                   for item in evidence_series.split(', ') if item.startswith('idx')]
            except (IndexError, ValueError):
                return False
        
        # Check each evidence item
        for evi_idx, evi_item in enumerate(evidence_items):
            try:
                if evi_idx < len(evidence_sections) and any(evidence_sections[evi_idx] == sec for sec in matching_sections):
                    if self._match_strings(evi_item, ent_value) or self._match_strings(evi_item, obs_itr):
                        # Check location similarity using model-based similarity
                        loc_sim_score, _, _ = LocationComparator.compare_locations(
                            current_std, modified_path_df, evi_item, obs_itr, obs_idx, self.status_analyzer
                        )
                        if loc_sim_score > 0.9:
                            return True
            except IndexError:
                continue
        
        return False
    
        
    def process_study(self, current_std: pd.DataFrame, previous_further_match_idx: List[int], 
                     report_type: str = 'section-level', further_analysis: bool = False) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Process a study by applying pathways to entities and updating evidence.
        
        Args:
            current_std: DataFrame containing the current study data
            previous_further_match_idx: List of previously matched indices
            report_type: Type of report ('section-level' or other)
            further_analysis: Whether to perform further analysis
            
        Returns:
            Tuple of (updated DataFrame, new entries DataFrame or None)
        """
        study_id = current_std['study_id'].iloc[0] if not current_std.empty else 'unknown'
        
        previous_std = None  # Initialize to None instead of creating a DataFrame
        # print(previous_further_match_idx)

        # store the current study as the previous study -> will need it for further processing
        if len(previous_further_match_idx) != 0:
            further_analysis = True # further_analysis flag is just for debugging
            previous_std = current_std.copy()
            # 'sent'가 NaN인 경우, 해당 행들만 처리
            # only invoke pathways for entities we have not processed before
            current_std = current_std[current_std['sent'].isna() & ~current_std['idx'].isin(previous_further_match_idx)]

        # rows for which a pathway was found and can be applied
        pathway_rows = current_std[current_std.pathway.notna()]
        # print(pathway_rows[["idx", "ent"]])
        
        if study_id == 's59542064':
            print(f"\n🔍 s59542064 Pathway Expansion:")
            print(f"Input rows: {len(current_std)}")
            print(f"Pathway rows: {len(pathway_rows)}")
            if not pathway_rows.empty:
                print(f"Pathway entities: {pathway_rows['ent'].unique()}")
            print(f"Previous std rows: {len(previous_std) if previous_std is not None else 0}")
        
        if pathway_rows.empty and previous_std is not None and not previous_std.empty:
            if study_id == 's59542064':
                print(f"Returning previous_std: {len(previous_std)} rows")
            return previous_std, None
        elif pathway_rows.empty and (previous_std is None or previous_std.empty):
            if study_id == 's59542064':
                print(f"Returning current_std: {len(current_std)} rows")
            return current_std, None
        
        # get status processor, entity names and indices
        status_processor = StatusProcessor()
        dx = pathway_rows['ent'].values
        dx_idx = pathway_rows['idx'].values
        study_max_idx = current_std['idx'].values # to decide where index of added rows should start
        
        # get pathway, status and location
        cur_pathway = pathway_rows['pathway'].values
        cur_status = pathway_rows['status'].apply(lambda x: str(x).split('|')[0]).values
        cur_loc = pathway_rows['location'].values

        new_idx, new_entries = int(max(study_max_idx)), []
        # iterate over each diagnosis and its pathway
        for idx, status_itr in enumerate(cur_status):
            pathway_str = str(cur_pathway[idx])

            # Update status based on report
            # rest of pathway string remains the same
            # if pathway cannot be applied (because diagnosis is TN/DN), then status will be None
            modified_path = re.sub(r'status: (dp|dn|tp|tn)', 
                                  lambda match: status_processor.switch_status(status_itr, match), 
                                  pathway_str) 

            # Add location information
            if not pd.isna(cur_loc[idx]):
                modified_path = LocationProcessor.process_paths(modified_path, cur_loc[idx])
            # print(modified_path)

            # Convert pathway to DataFrame
            # the pathway string is processed and the values are put in their respective columns
            # fixed columns: entity, cat, subcategory, obs, status
            # optional columns: location (if parent entity has specific location), morphology, distribution, etc (other relational attributes of observations)
            if isinstance(modified_path, str) and modified_path.startswith('['):
                modified_path_df = PathwayAnalyzer.analyze_pathway_with_index(eval(modified_path), dx[idx])
            elif isinstance(modified_path, list):
                modified_path_df = PathwayAnalyzer.analyze_pathway_with_index(modified_path, dx[idx])
            else:
                modified_path_df = PathwayAnalyzer.analyze_pathway_with_index([modified_path], dx[idx])
            # print(modified_path_df)

            # Rename columns for consistency with gold dataset column names
            modified_path_df.rename(columns={'obs': 'ent', 'loc': 'location'}, inplace=True)
            modified_path_df = modified_path_df[
                (modified_path_df['ent'].notna()) & 
                (modified_path_df['status'].notna()) & 
                (modified_path_df['status'] != 'None') # filter out rows where status is None, it means pathway could not be applied
            ].reset_index()

            modified_path_df_obs_values = modified_path_df['ent'].to_list()
            
            # create new entries for all the observations in this pathway
            if modified_path_df_obs_values:
                new_entries.extend(self._process_observations(
                    current_std, dx[idx], dx_idx[idx], modified_path_df, 
                    modified_path_df_obs_values, report_type
                ))

            # print("--------")

        # Create DataFrame from new entries and assign indices
        new_entry_df = pd.DataFrame(new_entries) if new_entries else pd.DataFrame()
        
        # Combine DataFrames based on further_analysis flag
        if further_analysis:
            if not new_entry_df.empty:
                new_entry_df['idx'] = range(new_idx+1, new_idx + len(new_entry_df)+1)
                current_std = pd.concat([previous_std, new_entry_df], ignore_index=True)
                if study_id == 's59542064':
                    print(f"\n🔗 s59542064 After concat (further_analysis=True):")
                    print(f"previous_std rows: {len(previous_std)}")
                    print(f"new_entry_df rows: {len(new_entry_df)}")
                    print(f"current_std rows after concat: {len(current_std)}")
                    print(f"current_std entities: {current_std['ent'].unique()}")
            else:
                # No new entries, return previous_std as is
                current_std = previous_std.copy()
                if study_id == 's59542064':
                    print(f"\n🔗 s59542064 No new entries (further_analysis=True):")
                    print(f"current_std rows: {len(current_std)}")
                    print(f"current_std entities: {current_std['ent'].unique()}")
        else:
            if not new_entry_df.empty:
                new_entry_df['idx'] = range(new_idx+1, new_idx + len(new_entry_df)+1)
                current_std = pd.concat([current_std, new_entry_df], ignore_index=True)
                if study_id == 's59542064':
                    print(f"\n🔗 s59542064 After concat (further_analysis=False):")
                    print(f"current_std rows after concat: {len(current_std)}")
                    print(f"current_std entities: {current_std['ent'].unique()}")
        
        # Update evidence only if we have new entries
        if not new_entry_df.empty:
            if study_id == 's59542064':
                print(f"\n📝 s59542064 Before _update_evidence:")
                print(f"current_std rows: {len(current_std)}")
                print(f"current_std entities: {current_std['ent'].unique()}")
            
            self._update_evidence(current_std, new_idx)
            
            if study_id == 's59542064':
                print(f"\n📝 s59542064 After _update_evidence:")
                print(f"current_std rows: {len(current_std)}")
                print(f"current_std entities: {current_std['ent'].unique()}")
            
        # Location column is already in the correct format
        # print(new_entry_df[["idx", "ent"]])

        # if study_id == 's59542064':
        #     print(f"\n✅ s59542064 Pathway Expansion Complete:")
        #     print(f"Final rows: {len(current_std)}")
        #     print(f"Final entities: {current_std['ent'].unique()}")
        #     if not new_entry_df.empty:
        #         print(f"New entries added: {len(new_entry_df)}")
        #         print(f"New entities: {new_entry_df['ent'].unique()}")
        #     input("Press Enter to continue after pathway expansion...")

        return current_std, new_entry_df
    
    def _process_observations(self, current_std: pd.DataFrame, dx: str, dx_idx: int, 
                             modified_path_df: pd.DataFrame, obs_values: List[str],
                             report_type: str) -> List[Dict]:
        """Process observations from the pathway and match with entities in the report."""
        new_entries = []
        
        # go over all new observations which should be added according to the pathway
        for obs_idx, obs_itr in enumerate(obs_values):
            no_match_list = []
            
            # loop over all current entities to see if there is a match with the new observation
            for ent_idx, ent_value in enumerate(current_std['ent'].values):
                if pd.isna(ent_value):
                    continue
                    
                # select the row for which we are applying the pathway (so containing the disease)
                # we will use it as a basis to fill in the new observation's row
                disease_row = current_std[
                    (current_std['ent'] == dx) & 
                    (current_std['idx'] == dx_idx)
                ].iloc[0].copy()
                
                # new observation should be associated to disease entity
                disease_row['associate'] = f"{disease_row['ent']}, idx{int(disease_row['idx'])}"
                evidence_series = disease_row['evidence'] # we don't update evidence yet (see later in code, _update_evidence)
                associate_series = disease_row['associate']
                
                # Check if the pathway observation matches with existing entity (using both ent and normed_ent from vocab)
                if self._match_entities_with_normed(obs_itr, ent_value):
                    # Check if this entity is already related to the disease through evidence
                    # This method already includes location similarity check internally
                    if self._is_entity_related_to_disease(disease_row, ent_value, obs_itr, current_std, modified_path_df, obs_idx, report_type):
                        # Entity is already related (same location), skip adding
                        break
                    else:
                        # Entity exists but not related (different location), allow pathway expansion
                        continue
            
            # If no match found after checking all entities, create a new row for the observation
            if ent_idx == len(current_std['ent'].values) - 1:
                # Clear columns based on report type
                if report_type == 'section-level':
                    columns_to_clear = [col for col in disease_row.index if col not in VALID_ATTR_CATEGORY2]
                else:
                    columns_to_clear = [col for col in disease_row.index if col not in REPORT_LEVEL_CATEGORY]
                
                # Preserve source prob before clearing and restore after
                source_prob = disease_row['prob'] if 'prob' in disease_row.index else np.nan
                disease_row[columns_to_clear] = np.nan
                if 'prob' in disease_row.index:
                    disease_row['prob'] = source_prob
                
                # Update matching row with modified values
                self._update_matching_row(disease_row, modified_path_df, obs_itr, obs_idx, new_entries)
                    
        return new_entries
    
    def _check_evidence(self, evidence_series: str, check_sections: List[str], 
                       ent_value: str, obs_itr: str, obs_idx: int, 
                       current_std: pd.DataFrame, modified_path_df: pd.DataFrame,
                       report_type: str) -> bool:
        """Check if the observation is already in evidence."""
        # check if new observation (obs_itr) or value of its matching row (ent_value) is already part of evidence of original disease row
        # only check for sections in check_sections
        # if it is, then we do not add the new observation as a row, since that would constitue redundant information

        if pd.notna(evidence_series) and evidence_series != '' and isinstance(evidence_series, str):
            evidence_items = [item for item in evidence_series.split(', ') if not item.startswith('idx')]
            
            # Get evidence sections
            try:
                if report_type == 'section-level': # only check rows in same section
                    evidence_section = [
                        current_std[current_std['idx'] == int(item.split('idx')[-1])]['section'].values[0] 
                        for item in evidence_series.split(', ') if item.startswith('idx')
                    ]
                else:
                    evidence_section = [ # only check rows in same diagnosis type (current or past)
                        current_std[current_std['idx'] == int(item.split('idx')[-1])]['DxTP'].values[0] 
                        for item in evidence_series.split(', ') if item.startswith('idx')
                    ]
            except IndexError as e:
                idx_value = next(
                    (int(item.split('idx')[-1]) for item in evidence_series.split(', ') 
                     if item.startswith('idx') and current_std[current_std['idx'] == int(item.split('idx')[-1])].empty), 
                    None
                )
                study_id = current_std['study_id'].iloc[0] if 'study_id' in current_std.columns else 'Unknown'
                print(f"Error: {e} for idx {idx_value} in study_id {study_id}")
                return False
            
            # Check each evidence item
            for evi_idx, evi_item in enumerate(evidence_items):
                loc_sim_score = 0
                
                try:
                    # check section alignment
                    if any(evidence_section[evi_idx] == sec for sec in check_sections): 
                        # check if evidence item (evi_item) matches with new observation value (obs_itr) or its matching row value (ent_value)
                        if self._match_strings(evi_item, ent_value) or self._match_strings(evi_item, obs_itr):
                            # check if locations of evi_item and obs_itr are sufficiently similar, by checking current_std (study rows) and modified_path_df (pathway observations)
                            loc_sim_score, _, _ = LocationComparator.compare_locations(
                                current_std, modified_path_df, evi_item, obs_itr, obs_idx
                            )
                            if loc_sim_score > 0.6:
                                return True
                except IndexError:
                    continue
                    
        return False
    
    def _update_matching_row(self, new_row: pd.Series, modified_path_df: pd.DataFrame, 
                            obs_itr: str, obs_idx: int, new_entries: List[Dict]) -> None:
        """Update the new row with values from the pathway (cat, location, remaining attributes)."""
        # obs_row is the row from the pathway df which contains attributes for obs_itr (current pathway observation for which we are constructing the row)
        # new_row is the new row we are constructing, starting from the parent disease row which already has some attributes filled in
        obs_row = modified_path_df.loc[(modified_path_df['ent'] == obs_itr) & (modified_path_df.index == obs_idx)]
        
        for col in obs_row.columns:
            if pd.notna(obs_row[col]).any() and col in new_row.index:
                new_value = obs_row[col].values[0]
                existing_value = new_row[col] # value which is already present in this column from parent disease
                
                # Check conditions for updating values
                # Normalize category text for robust comparison (handles 'patient' vs 'patient info.')
                cat_series = obs_row['cat'].astype(str).str.lower().str.strip()
                cat_series = cat_series.str.replace(r'\bpatient\b\.?$', 'patient info.', regex=True)
                condition_1 = cat_series.isin([c.lower() for c in ATTRIBUTE_NOT_ALLOWED_CATEGORY]).all() # patient info. should not inherit parent clinical attrs
                condition_2 = obs_row[
                    (obs_row['ent'].isin(self.except_obs_list)) & 
                    (obs_row['status'].isin(['dn', 'tn'])) # negative observations (DN pulmonary marking for pneumothorax, DN volume loss for consolidation) should not receive parent attributes
                ].any().any()
                # Patient info. (or other ATTRIBUTE_NOT_ALLOWED_CATEGORY) must not have location
                if condition_1 and 'location' in new_row.index:
                    new_row['location'] = np.nan
                
                # if there is already a value in the column we are updating, we add the new value after the column
                # for example, this happens for location
                # we only do this if the attribute type is allowed (both category and subcategory)
                if pd.notna(existing_value) and not obs_row['subcategory'].isin(ATTRIBUTE_NOT_ALLOWED_SUBCATEGORY).all() and not obs_row['cat'].isin(ATTRIBUTE_NOT_ALLOWED_CATEGORY).all():
                    new_row[col] = f"{existing_value}, {new_value}"
                # if the column is originally empty, or the category/subcategory of the observation is not allowed, we advance to this check
                # condition_1: the category of the observation we are adding is in the not allowed list
                # condition_2: the observation is negative (like in pulmonary marking, or volume loss)
                # if either of these conditions are true, we clear all the parent attributes in CLINICAL_ATTR_CAT (like morphology, distribution, etc) and only add the observation's new attributes
                elif condition_1 or condition_2:
                    columns_to_clear = [col for col in new_row.index if col in CLINICAL_ATTR_CAT]
                    new_row[columns_to_clear] = np.nan
                    new_row[col] = new_value
                # in all other cases, we just set the new attribute column to correspond to the column in the pathway, keeping all the parent attributes as they are
                else:
                    new_row[col] = new_value
        
        # If expanded observation's pathway status is DN, force prob to 0
        try:
            obs_status = str(obs_row['status'].values[0]).strip().lower()
            if 'prob' in new_row.index and obs_status == 'dn':
                new_row['prob'] = 0.0
        except Exception:
            pass

        # Add to new entries if not already processed
        row_string = row_to_string(new_row.to_dict())
        if row_string not in self.processed_pairs:
            new_entries.append(new_row.to_dict())
            self.processed_pairs.append(row_string)
    
    def _update_evidence(self, current_std: pd.DataFrame, new_idx: int) -> None:
        """Update evidence for newly added entries."""
        sent_nan_rows = current_std[current_std['idx'] > new_idx] # newly added entries have empty sent columns
        
        for _, row in sent_nan_rows.iterrows():
            ent_value, idx_value = row['ent'], row['idx']
            # the associate column already contains the parent disease that lead to the construction of the row, so use this to fill in the evidence column
            target_ent, target_idx = row['associate'].split(', idx') 
            
            # we need to add the new entity as evidence in the parent disease row that led to the application of the pathway
            target_row = current_std[(current_std['ent'] == target_ent) & (current_std['idx'] == int(target_idx))]
            
            if not target_row.empty:
                # 주피터 노트북과 동일한 로직: 모든 새 엔티티를 evidence에 추가
                evidence_value = target_row['evidence'].iloc[0]
                new_evidence = f"{evidence_value}, {ent_value}, idx{int(idx_value)}" if not pd.isna(evidence_value) else f"{ent_value}, idx{int(idx_value)}"
                
                # evidence 업데이트
                current_std.loc[current_std['idx'] == int(target_idx), 'evidence'] = new_evidence
                
class TextProcessor:
    """Utility class for text processing."""
    
    @staticmethod
    def clean_text(text):
        """Clean text by removing excess whitespace and normalizing commas."""
        if not isinstance(text, str):
            return text
            
        # Remove leading/trailing spaces
        cleaned = text.strip()
        
        # Replace multiple spaces with a single space
        cleaned = ' '.join(cleaned.split())
        
        # Fix comma issues (remove leading/trailing commas, replace multiple commas)
        cleaned = cleaned.strip(',')
        while ',,' in cleaned:
            cleaned = cleaned.replace(',,', ',')
        
        # Fix spaces around commas
        cleaned = cleaned.replace(' , ', ', ')
        cleaned = cleaned.replace(', ,', ',')
        
        return cleaned
    
    @staticmethod
    def remove_duplicates_and_sort(text):
        """Remove duplicate terms and sort alphabetically."""
        if not isinstance(text, str):
            return text
        parts = [part.strip() for part in text.split(',') if part.strip()]
        unique_parts = sorted(set(parts), key=str.lower)
        return ', '.join(unique_parts)
    
    @staticmethod
    def replace_terms(text, mapping):
        """Replace terms according to mapping dictionary."""
        if not isinstance(text, str):
            return text
        # Special case handling
        text = text.replace('CHEST (PA AND LAT)', 'PA, LATERAL')
        
        # Apply general mappings
        for key, queries in mapping.items():
            for query in queries:
                text = re.sub(rf'\b{query}\b', key, text)
        return text

class ConflictDetector:
    """
    Detect and process conflicts in medical report data
    
    This class finds conflicts in medical report data:
    - For same study_id & ent, different statuses
    - Conflicts in evidence or associate
    """
    
    def __init__(self, output_dir: str = './dx_pathway/'):
        """
        Initialize ConflictDetector
        
        Args:
            output_dir: Directory path to save result files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    @staticmethod
    def match_loc(obs_value: Any, ent_value: Any) -> bool:
        """
        Compare location values
        
        Args:
            obs_value: First value to compare
            ent_value: Second value to compare
            
        Returns:
            bool: True if the values match, False otherwise
        """
        # Handle NaN values
        if pd.isna(obs_value) and pd.isna(ent_value):
            return True

        # Check if both values are strings
        if isinstance(obs_value, str) and isinstance(ent_value, str):
            # Simple string comparison
            return obs_value.strip() == ent_value.strip()

        return False
    
    def process_section_conflicts(self, df: pd.DataFrame, study_id: Union[int, str], 
                                 report_type: str = 'section-level') -> pd.DataFrame:
        """
        Process section (or DxTP) and location matching for given study_id,
        and return rows where same entity has conflicting statuses.
        Additionally, classify conflict types into three buckets:
          - original_vs_original: conflicts only among grounded rows (sent present)
          - original_vs_expansion: conflicts between grounded and expansion rows
          - expansion_vs_expansion: conflicts only among expansion rows (sent is NaN)
        
        Args:
            df: DataFrame containing all data
            study_id: Value of study_id to process
            report_type: 'section-level' or 'report-level'
            
        Returns:
            pd.DataFrame: DataFrame containing rows with conflicts and conflict_type column
        """
        # Filter data for given study_id
        study_data = df[df['study_id'] == study_id].copy()
        
        # Process section by section
        if report_type == 'section-level':
            sections = study_data['section'].unique()
        else:
            sections = study_data['DxTP'].unique()
            
        duplicated_rows = []
        
        for section in sections:
            if report_type == 'section-level':
                report_section = study_data[study_data['section'] == section]
            else:
                report_section = study_data[study_data['DxTP'] == section]
                
            # Process each location
            for loc_itr in report_section['location'].unique():
                # Find rows where location matches
                matched_rows = report_section[report_section['location'].apply(
                    lambda x: self.match_loc(x, loc_itr)
                )]
                
                # Check status conflicts for each entity
                for ent in matched_rows['ent'].unique():
                    ent_rows = matched_rows[matched_rows['ent'] == ent]
                    if len(ent_rows) > 1:
                        # Normalize to primary status token (before '|')
                        primary_status = ent_rows['status'].apply(
                            lambda x: x.split('|')[0] if pd.notna(x) and '|' in str(x) else x
                        )
                        # Determine polarity presence
                        has_pos = primary_status.isin(['dp', 'tp']).any()
                        has_neg = primary_status.isin(['dn', 'tn']).any()
                        has_definitive = primary_status.isin(['dp', 'dn']).any()
                        has_tentative = primary_status.isin(['tp', 'tn']).any()
                        
                        # original vs expansion 구분
                        original_rows = ent_rows[ent_rows['sent'].notna()]
                        expansion_rows = ent_rows[ent_rows['sent'].isna()]
                        
                        # 충돌 타입 결정 (우선순위 기반)
                        conflict_type = None
                        conflict_source = None
                        
                        # 1. P vs N 충돌 (최우선)
                        if has_pos and has_neg:
                            conflict_type = 'polarity_conflict'
                            
                            # 충돌 소스 구분
                            if not original_rows.empty and not expansion_rows.empty:
                                conflict_source = 'original_vs_expansion'
                            elif not original_rows.empty and expansion_rows.empty:
                                conflict_source = 'original_vs_original'
                            elif original_rows.empty and not expansion_rows.empty:
                                conflict_source = 'expansion_vs_expansion'
                        
                        # 2. D vs T 충돌 (같은 polarity 내에서만)
                        elif has_pos and has_definitive and has_tentative:
                            conflict_type = 'certainty_conflict_positive'
                            
                            # 충돌 소스 구분
                            if not original_rows.empty and not expansion_rows.empty:
                                conflict_source = 'original_vs_expansion'
                            elif not original_rows.empty and expansion_rows.empty:
                                conflict_source = 'original_vs_original'
                            elif original_rows.empty and not expansion_rows.empty:
                                conflict_source = 'expansion_vs_expansion'
                                
                        elif has_neg and has_definitive and has_tentative:
                            conflict_type = 'certainty_conflict_negative'
                            
                            # 충돌 소스 구분
                            if not original_rows.empty and not expansion_rows.empty:
                                conflict_source = 'original_vs_expansion'
                            elif not original_rows.empty and expansion_rows.empty:
                                conflict_source = 'original_vs_original'
                            elif original_rows.empty and not expansion_rows.empty:
                                conflict_source = 'expansion_vs_expansion'
                        
                        # 3. 같은 status 중복 (dp, dn 제외)
                        else:
                            status_counts = primary_status.value_counts()
                            for status, count in status_counts.items():
                                if count > 1 and status in ['tp', 'tn']:
                                    conflict_type = f'duplicate_{status.lower()}'
                                    
                                    # 충돌 소스 구분
                                    if not original_rows.empty and not expansion_rows.empty:
                                        conflict_source = 'original_vs_expansion'
                                    elif not original_rows.empty and expansion_rows.empty:
                                        conflict_source = 'original_vs_original'
                                    elif original_rows.empty and not expansion_rows.empty:
                                        conflict_source = 'expansion_vs_expansion'
                                    break
                        
                        # 충돌이 감지된 경우 처리
                        if conflict_type:
                            # Detection-time skip for lungs polarity conflicts among originals
                            if (conflict_type == 'polarity_conflict' and 
                                conflict_source == 'original_vs_original'):
                                try:
                                    ent_l = str(ent).lower()
                                except Exception:
                                    ent_l = ''
                                if ('lung' in ent_l) or ('lungs' in ent_l):
                                    continue
                            temp = ent_rows.copy()
                            temp['conflict_type'] = conflict_type
                            temp['conflict_source'] = conflict_source
                            temp['conflict_scope'] = 'section' if report_type == 'section-level' else 'DxTP'
                            duplicated_rows.append(temp)
                            
        if duplicated_rows:
            return pd.concat(duplicated_rows, ignore_index=True)
        return pd.DataFrame()
    
    def find_all_conflicts(self, df: pd.DataFrame, report_type: str = 'section-level') -> pd.DataFrame:
        """
        Find conflicts for all study_id
        
        Args:
            df: DataFrame containing all data
            report_type: 'section-level' or 'report-level'
            
        Returns:
            pd.DataFrame: DataFrame containing all conflicts
        """
        all_duplicated_dfs = []
        
        # Process all study_id
        study_ids = df['study_id'].unique()
        
        for study_id in study_ids:
            study_duplicated_df = self.process_section_conflicts(df, study_id, report_type)
            if not study_duplicated_df.empty:
                all_duplicated_dfs.append(study_duplicated_df)
        
        if all_duplicated_dfs:
            return pd.concat(all_duplicated_dfs, ignore_index=True)
        return pd.DataFrame()
    
    def clean_object_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean object type columns in DataFrame
        
        Args:
            df: DataFrame to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        df_copy = df.copy()
        for col in df_copy.columns:
            if df_copy[col].dtype == 'object':
                df_copy[col] = df_copy[col].apply(lambda x: str(x) if isinstance(x, list) else x)
        return df_copy
    
    def process_and_save(self, section_df_path: str, report_df_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process and save section and report level data
        
        Args:
            section_df_path: Path to section level data file, can be None
            report_df_path: Path to report level data file, can be None
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (section level conflicts, report level conflicts) DataFrame
        """
        # Initialize empty DataFrames
        section_issue_df = pd.DataFrame()
        report_issue_df = pd.DataFrame()
        
        # Process section level data if path exists
        if section_df_path is not None:
            section_level_out = pd.read_csv(section_df_path)
            section_issue_df = self.find_all_conflicts(section_level_out, report_type='section-level')
            section_issue_df = self.clean_object_columns(section_issue_df)
            section_issue_df = section_issue_df.drop_duplicates()
        
        # Process report level data if path exists
        if report_df_path is not None:
            report_level_out = pd.read_csv(report_df_path)
            report_issue_df = self.find_all_conflicts(report_level_out, report_type='report-level')
            report_issue_df = self.clean_object_columns(report_issue_df)
            report_issue_df = report_issue_df.drop_duplicates()
            
        return section_issue_df, report_issue_df

class ConflictResolver:
    """
    Class for resolving conflicts in medical report data
    
    This class provides the following functionality:
    - Resolving status conflicts
    - Cleaning evidence and associate fields
    - Removing conflicting rows
    - Updating report data
    """
    
    def __init__(self, output_dir: str = './output/'):
        """
        Initialize ConflictResolver
        
        Args:
            output_dir: Directory path to save result files
        """
        self.output_dir = output_dir        
        # Status classifications
        self.positive_statuses = ['dp', 'tp']
        self.negative_statuses = ['dn', 'tn']
        
        # Define attribute columns
        self.attribute_columns = [
            'morphology', 'distribution', 'measurement', 'severity', 
            'comparison', 'onset', 'no change', 'improved', 'worsened', 
            'placement', 'past hx', 'other source', 'assessment limitations'
        ]
    
    def _parse_pairs(self, value: Any):
        """
        Parse comma-separated "entity, idxN" sequences into normalized pair tuples.
        Returns a sorted tuple of unique lowercased pairs; empty when value is NaN/blank.
        """
        if value is None:
            return tuple()
        try:
            s = str(value)
        except Exception:
            return tuple()
        if s.strip() == '' or s.strip().lower() == 'nan':
            return tuple()
        items = [t.strip().lower() for t in s.split(',') if t.strip() != '']
        pairs = []
        i = 0
        while i + 1 < len(items):
            left = items[i]
            right = items[i+1]
            pairs.append((left, right))
            i += 2
        # sort and de-duplicate
        return tuple(sorted(set(pairs)))

    def _eq_pairs(self, a: Any, b: Any) -> bool:
        """Equality after normalizing to pair tuples."""
        return self._parse_pairs(a) == self._parse_pairs(b)

    @staticmethod
    def clean_column(value: Any, target_value: str) -> Any:
        """
        Remove specific value from a column value
        
        Args:
            value: Original value
            target_value: Value to remove
            
        Returns:
            Cleaned value or NaN
        """
        if pd.notna(value):
            # Remove target_value
            updated_value = value.replace(target_value, '').strip()
            # Remove unnecessary ', ' and trim whitespace
            updated_value = updated_value.replace(', ,', ',').strip(', ')
            return updated_value if updated_value else np.nan  # Return NaN if empty
        return value
    
    def delete_conflicts(self, final_out: pd.DataFrame, abandoned_df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove conflicting rows and clean related fields
        
        Args:
            final_out: Original DataFrame
            abandoned_df: DataFrame containing rows to be removed
            
        Returns:
            DataFrame with conflicts resolved
        """
        if abandoned_df.empty:
            return final_out
            
        final_out = final_out.copy()
        abandoned_df = abandoned_df.copy()
        
        final_out['study_id'] = final_out['study_id'].astype(str)
        final_out['idx'] = final_out['idx'].astype(int)
        abandoned_df['study_id'] = abandoned_df['study_id'].astype(str)
        abandoned_df['idx'] = abandoned_df['idx'].astype(int)
            
        matching_rows = final_out[['study_id', 'idx']].apply(tuple, axis=1).isin(
            abandoned_df[['study_id', 'idx']].apply(tuple, axis=1)
        )
        
        if matching_rows.empty:
            raise Exception("No matching rows found")
        
        for index, row in abandoned_df.iterrows():
            study_id = row['study_id']
            ent_idx_value = f"{row['ent']}, idx{row['idx']}"
            
            # Filter rows with the same study_id
            same_study_rows = final_out[final_out['study_id'] == study_id]
            
            # Clean evidence and associate
            for col in ['evidence', 'associate']:
                conflicting_rows = same_study_rows[
                    same_study_rows[col].apply(lambda x: pd.notna(x) and ent_idx_value in str(x))
                ]
                
                if not conflicting_rows.empty:
                    for idx, conf_row in conflicting_rows.iterrows():
                        updated_value = self.clean_column(conf_row[col], ent_idx_value)
                        final_out.loc[idx, col] = updated_value
        
        # Remove conflicting rows
        if 'study_id' in abandoned_df.columns and 'idx' in abandoned_df.columns:
            final_out_filtered = final_out[~matching_rows]
            print(f"{len(final_out)} -> after filtering - final_out rows:", len(final_out_filtered))
            print("\n")
        else:
            print("Warning: Required columns not found in abandoned_df")
            final_out_filtered = final_out
            
        return final_out_filtered
    
    def process_further_changes(self, final_out: pd.DataFrame, 
                              further_delete_df: pd.DataFrame, 
                              further_change_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process additional changes
        
        Args:
            final_out: Original DataFrame
            further_delete_df: DataFrame containing rows to be additionally deleted
            further_change_df: DataFrame containing rows to be additionally changed
            
        Returns:
            DataFrame with changes applied
        """
        final_out = final_out.copy()
        
        # Process further_delete_df
        for _, row in further_delete_df.iterrows():
            evidence_items = row['evidence'].split(',')
            evidence_pairs = [(evidence_items[i].strip(), evidence_items[i+1].strip()) 
                             for i in range(0, len(evidence_items), 2)]
            for ent, idx in evidence_pairs:
                if 'idx' in idx:
                    idx = idx.split('idx')[1].strip()
                    mask = ((final_out['study_id'] == row['study_id']) & 
                           (final_out['ent'] == ent) & 
                           (final_out['idx'] == int(idx)))
                    final_out = final_out[~mask]

        # Process further_change_df
        for _, row in further_change_df.iterrows():
            evidence_items = row['evidence'].split(',')
            evidence_pairs = [(evidence_items[i].strip(), evidence_items[i+1].strip()) 
                             for i in range(0, len(evidence_items), 2)]
            for ent, idx in evidence_pairs:
                if 'idx' in idx:
                    idx = idx.split('idx')[1].strip()
                    mask = ((final_out['study_id'] == row['study_id']) & 
                           (final_out['ent'] == ent) & 
                           (final_out['idx'] == int(idx)))
                    final_out.loc[mask, 'status'] = row['ori_status']

        return final_out

    def _print_conflict_debug(self, prefix, df):
        # df는 비교 대상 행들의 작은 DataFrame (원본/확장/결합 각각)
        cols = ['ent','study_id','section','location','evidence','associate']
        printable = df[cols] if all(c in df.columns for c in cols) else df
        print(f"\n[{prefix}]")
        try:
            print(printable.to_dict(orient='records'))
        except Exception:
            print(printable.head(3))
    
    def resolve_status_conflicts(self, issue_df: pd.DataFrame, 
                               report_type: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Resolve status conflicts
        
        Args:
            issue_df: DataFrame containing conflicts
            report_type: 'section-level' or 'report-level'
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (resolved rows, rows to be removed)
        """
        subgroup_categories = ['ent', 'study_id', 'location', 'conflict_type', 'conflict_source']
        
        # Log for specific study
        s59542064_conflicts = issue_df[issue_df['study_id'] == 's59542064'] if not issue_df.empty else pd.DataFrame()
        if not s59542064_conflicts.empty:
            print(f"\n🔧 s59542064 Conflict Resolution:")
            print(f"Conflicts to resolve: {len(s59542064_conflicts)}")
            print(f"Conflicting entities: {s59542064_conflicts['ent'].unique()}")
        
        resolved_rows = []
        abandoned_list = []

        if report_type == 'section-level':
            subgroup_categories.append('section')
        else:
            subgroup_categories.append('DxTP')

        issue_df['location'] = issue_df['location'].fillna('')
        
        subgroups = issue_df.groupby(subgroup_categories)
        
        for keys, group in subgroups:
            group = group.copy()
            conflict_type = group['conflict_type'].iloc[0] if 'conflict_type' in group.columns else 'unknown'
            conflict_source = group['conflict_source'].iloc[0] if 'conflict_source' in group.columns else 'unknown'

            original_rows = group[group['sent'].notna()]
            expanded_rows = group[group['sent'].isna()]

            # Helper to pick rows by prob (max or min) with sensible fallback
            def _pick_by_prob(rows: pd.DataFrame, mode: str = 'max') -> Tuple[pd.DataFrame, pd.DataFrame]:
                if rows.empty:
                    return rows, rows
                if 'prob' in rows.columns and rows['prob'].notna().any():
                    pick_idx = rows['prob'].idxmax() if mode == 'max' else rows['prob'].idxmin()
                    picked = rows.loc[[pick_idx]]
                    dropped = rows.drop(index=pick_idx)
                else:
                    picked = rows.iloc[[0]]
                    dropped = rows.iloc[1:]
                return picked, dropped

            # 1) original_vs_expansion: always follow original
            if conflict_source == 'original_vs_expansion':
                if original_rows.empty:
                    # Fallback: if no original, resolve within expansions using rules below
                    pass
                else:
                    combined_row = original_rows.copy()
                    for col in ['evidence', 'associate']:
                        unique_values = group[col].astype(str).str.lower().unique()
                        filtered_values = [x.strip() for x in filter(lambda x: x != 'nan' and x != '', unique_values)]
                        filtered_values = [x for x in filtered_values if x]
                        combined_row[col] = ', '.join(filtered_values) if filtered_values else ''

                    if ((not original_rows['evidence'].isna().all() or not original_rows['associate'].isna().all()) or 
                        (not expanded_rows['evidence'].isna().all() or not expanded_rows['associate'].isna().all())):
                        orig_ev = original_rows['evidence'].iloc[0] if not original_rows.empty else None
                        orig_as = original_rows['associate'].iloc[0] if not original_rows.empty else None
                        exp_ev = expanded_rows['evidence'].iloc[0] if not expanded_rows.empty else None
                        exp_as = expanded_rows['associate'].iloc[0] if not expanded_rows.empty else None
                        cmb_ev = combined_row['evidence'].iloc[0] if not combined_row.empty else None
                        cmb_as = combined_row['associate'].iloc[0] if not combined_row.empty else None

                        cmb_ev_set = set(self._parse_pairs(cmb_ev))
                        cmb_as_set = set(self._parse_pairs(cmb_as))
                        orig_ev_set = set(self._parse_pairs(orig_ev))
                        orig_as_set = set(self._parse_pairs(orig_as))
                        exp_ev_set = set(self._parse_pairs(exp_ev))
                        exp_as_set = set(self._parse_pairs(exp_as))

                        union_ev = orig_ev_set | exp_ev_set
                        union_as = orig_as_set | exp_as_set

                        ev_reflected = (len(union_ev) == 0 and len(cmb_ev_set) == 0) or cmb_ev_set.issuperset(union_ev)
                        as_reflected = (len(union_as) == 0 and len(cmb_as_set) == 0) or cmb_as_set.issuperset(union_as)

                        if not (ev_reflected and as_reflected):
                            print("\n===== NOT REFLECTED DEBUG =====")
                            self._print_conflict_debug("ORIGINAL", original_rows)
                            self._print_conflict_debug("EXPANDED", expanded_rows)
                            self._print_conflict_debug("COMBINED", combined_row)
                            print(f"orig_ev={self._parse_pairs(orig_ev)}\nexp_ev={self._parse_pairs(exp_ev)}\ncmb_ev={self._parse_pairs(cmb_ev)}")
                            print(f"orig_as={self._parse_pairs(orig_as)}\nexp_as={self._parse_pairs(exp_as)}\ncmb_as={self._parse_pairs(cmb_as)}")
                            print(f"union_ev={sorted(union_ev)} union_as={sorted(union_as)}")
                            raise Exception("Not reflected")

                    if not isinstance(combined_row, pd.DataFrame):
                        combined_row = pd.DataFrame([combined_row])
                    resolved_rows.append(combined_row)
                    if not expanded_rows.empty:
                        abandoned_list.append(expanded_rows)
                    continue

            # 2) original_vs_original: resolve among originals with prob-aware rules
            if conflict_source == 'original_vs_original':
                rows = original_rows
                if rows.empty:
                    continue
                # Skip suspicious polarity conflicts for broad entity 'lung'
                ent_val = str(group['ent'].iloc[0]).lower() if 'ent' in group.columns and pd.notna(group['ent'].iloc[0]) else ''
                if conflict_type == 'polarity_conflict' and ('lung' in ent_val or 'lungs' in ent_val):
                    # Likely false-positive due to broad term; ignore this conflict group
                    continue
                statuses = rows['status'].apply(lambda s: str(s).split('|')[0] if pd.notna(s) else s)
                has_dp = (statuses == 'dp').any()
                has_dn = (statuses == 'dn').any()
                has_tp = (statuses == 'tp').any()
                has_tn = (statuses == 'tn').any()

                picked = None
                dropped = pd.DataFrame()

                if conflict_type == 'polarity_conflict':
                    # If only dp & dn present, delete both
                    if has_dp and has_dn and not (has_tp or has_tn):
                        temp = rows.copy()
                        temp['resolution_note'] = 'original_vs_original polarity dp<->dn: removed both'
                        abandoned_list.append(temp)
                        continue
                    # Otherwise pick prob max across both polarities (fallback: P first)
                    if 'prob' in rows.columns and rows['prob'].notna().any():
                        picked, dropped = _pick_by_prob(rows, 'max')
                    else:
                        pos = rows[rows['status'].isin(self.positive_statuses)]
                        if not pos.empty:
                            picked = pos.iloc[[0]]
                            dropped = rows.drop(index=picked.index)
                        else:
                            picked = rows.iloc[[0]]
                            dropped = rows.iloc[1:]

                elif conflict_type == 'certainty_conflict_positive':
                    if 'prob' in rows.columns and rows['prob'].notna().any():
                        picked, dropped = _pick_by_prob(rows, 'max')
                    else:
                        dp_rows = rows[rows['status'] == 'dp']
                        if not dp_rows.empty:
                            picked = dp_rows.iloc[[0]]
                            dropped = rows.drop(index=picked.index)
                        else:
                            picked = rows.iloc[[0]]
                            dropped = rows.iloc[1:]

                elif conflict_type == 'certainty_conflict_negative':
                    if 'prob' in rows.columns and rows['prob'].notna().any():
                        picked, dropped = _pick_by_prob(rows, 'max')
                    else:
                        dn_rows = rows[rows['status'] == 'dn']
                        if not dn_rows.empty:
                            picked = dn_rows.iloc[[0]]
                            dropped = rows.drop(index=picked.index)
                        else:
                            picked = rows.iloc[[0]]
                            dropped = rows.iloc[1:]

                elif conflict_type == 'duplicate_tp':
                    picked, dropped = _pick_by_prob(rows, 'max')

                elif conflict_type == 'duplicate_tn':
                    picked, dropped = _pick_by_prob(rows, 'max')

                else:
                    temp = rows.copy()
                    temp['resolution_note'] = 'original_vs_original: removed both (unknown type)'
                    abandoned_list.append(temp)
                    continue

                if picked is not None and not picked.empty:
                    resolved_rows.append(picked)
                if not dropped.empty:
                    dropped = dropped.copy()
                    dropped['resolution_note'] = f'original_vs_original: dropped due to {conflict_type} (prob-aware)'
                    abandoned_list.append(dropped)
                continue

            # 3) expansion_vs_expansion: resolve among expansions with prob-aware rules
            if conflict_source == 'expansion_vs_expansion':
                rows = expanded_rows
                if rows.empty:
                    continue
                statuses = rows['status'].apply(lambda s: str(s).split('|')[0] if pd.notna(s) else s)
                has_dp = (statuses == 'dp').any()
                has_dn = (statuses == 'dn').any()
                has_tp = (statuses == 'tp').any()
                has_tn = (statuses == 'tn').any()

                picked = None
                dropped = pd.DataFrame()

                if conflict_type == 'polarity_conflict':
                    # If only dp & dn present, delete both
                    if has_dp and has_dn and not (has_tp or has_tn):
                        temp = rows.copy()
                        temp['resolution_note'] = 'expansion_vs_expansion polarity dp<->dn: removed both'
                        abandoned_list.append(temp)
                        continue
                    # Otherwise pick prob max across both polarities (fallback: P first)
                    picked, dropped = _pick_by_prob(rows, 'max')

                elif conflict_type == 'certainty_conflict_positive':
                    picked, dropped = _pick_by_prob(rows, 'max')

                elif conflict_type == 'certainty_conflict_negative':
                    picked, dropped = _pick_by_prob(rows, 'max')

                elif conflict_type == 'duplicate_tp':
                    picked, dropped = _pick_by_prob(rows, 'max')

                elif conflict_type == 'duplicate_tn':
                    picked, dropped = _pick_by_prob(rows, 'max')

                else:
                    # Unknown: drop all to be safe
                    temp = rows.copy()
                    temp['resolution_note'] = 'expansion_vs_expansion: removed all (unknown type)'
                    abandoned_list.append(temp)
                    continue

                if picked is not None and not picked.empty:
                    resolved_rows.append(picked)
                if not dropped.empty:
                    dropped = dropped.copy()
                    dropped['resolution_note'] = f'expansion_vs_expansion: dropped due to {conflict_type} (prob-aware)'
                    abandoned_list.append(dropped)
                continue

            # Fallback for unexpected cases
            print("ERROR!!", expanded_rows['study_id'].unique() if not expanded_rows.empty else "No expanded rows")
        
        if resolved_rows:  
            resolved_df = pd.concat(resolved_rows, ignore_index=True)
        else: 
            resolved_df = pd.DataFrame()  

        if abandoned_list:  
            abandoned_df = pd.concat(abandoned_list, ignore_index=True)
        else: 
            abandoned_df = pd.DataFrame()  

        return resolved_df, abandoned_df
    
    def update_report_level_df(self, report_level_df: pd.DataFrame, 
                             target_rows: pd.DataFrame) -> pd.DataFrame:
        """
        Update report level DataFrame
        
        Args:
            report_level_df: Original DataFrame
            target_rows: DataFrame containing rows to update
            
        Returns:
            Updated DataFrame
        """
        # Create a copy of the input DataFrame to avoid modifying the original
        updated_df = report_level_df.copy()
        target_rows = target_rows.copy()
        
        target_rows['location'] = target_rows['location'].fillna('')
        updated_df['location'] = updated_df['location'].fillna('')

        # Process each row in target_rows
        for idx, target_row in target_rows.iterrows():
            mask = updated_df.apply(lambda x: (
                (x['study_id'] == target_row['study_id']) & 
                (x['ent'] == target_row['ent']) & 
                (x['idx'] == target_row['idx']) & 
                (x['location'] == target_row['location'])
            ), axis=1)
            # Update matched rows with target row values
            for col in updated_df.columns:
                updated_df.loc[mask, col] = target_row[col]
        return updated_df
    
    def filter_invalid_evidence_associate(self, matching_dataframe: pd.DataFrame, 
                                        report_level_df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter invalid evidence and associate
        
        Args:
            matching_dataframe: Original DataFrame
            report_level_df: Filtered report level DataFrame
            
        Returns:
            Filtered DataFrame
        """
        # Basic implementation - should be modified according to actual requirements
        # goal for the function: fix evidence index and associate index mistakes that result from deleting duplicate rows
        return report_level_df
    
    def process_conflicts(self, section_level_out: pd.DataFrame, report_level_out: pd.DataFrame, 
                        section_issue_df: pd.DataFrame, report_issue_df: pd.DataFrame,
                        save_output: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Main function to process all conflicts
        
        Args:
            section_level_out: Path to section level data file
            report_level_out: Path to report level data file
            section_issue_df: Path to section level issues file
            report_issue_df: Path to report level issues file
            save_output: Whether to save output files
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (processed section level data, processed report level data)
        """
        # Initialize update holders to originals (in case there are no resolved rows)
        section_level_updated_out = section_level_out.copy() if section_level_out is not None else pd.DataFrame()
        report_level_updated_out = report_level_out.copy() if report_level_out is not None else pd.DataFrame()
        final_section_level_out = section_level_updated_out.copy()
        final_report_level_out = report_level_updated_out.copy()

        # Process section level issues if exists
        if section_level_out is not None and section_issue_df is not None:
            section_resolved_df, section_abandoned_df = self.resolve_status_conflicts(
                section_issue_df, report_type='section-level')
            
            print(f"Section-level: change: {len(section_resolved_df)} delete: {len(section_abandoned_df)}, "
                  f"{len(section_issue_df)-len(section_abandoned_df)-len(section_resolved_df)} left")
            
            # Update section level data
            if not section_resolved_df.empty:
                section_level_updated_out = self.update_report_level_df(section_level_out, section_resolved_df)
            
            # Delete conflicts from section level data
            if not section_abandoned_df.empty:
                deleted_section_df = self.delete_conflicts(section_level_updated_out, section_abandoned_df)
                final_section_level_out = self.filter_invalid_evidence_associate(
                    matching_dataframe=section_level_updated_out, 
                    report_level_df=deleted_section_df)
        
        # Process report level issues if exists
        if report_level_out is not None and report_issue_df is not None:
            report_resolved_df, report_abandoned_df = self.resolve_status_conflicts(
                report_issue_df, report_type='report-level')
            
            print(f"Report-level: change: {len(report_resolved_df)} delete: {len(report_abandoned_df)}, "
                  f"{len(report_issue_df)-len(report_abandoned_df)-len(report_resolved_df)} left")
            
            # Update report level data
            if not report_resolved_df.empty:
                report_level_updated_out = self.update_report_level_df(report_level_out, report_resolved_df)
                
            # Delete conflicts from report level data
            if not report_abandoned_df.empty:
                deleted_report_df = self.delete_conflicts(report_level_updated_out, report_abandoned_df)
                final_report_level_out = self.filter_invalid_evidence_associate(
                    matching_dataframe=report_level_updated_out, 
                    report_level_df=deleted_report_df)

        return final_section_level_out, final_report_level_out

def filter_invalid_evidence_associate(matching_dataframe=None, report_level_df=None):
    def process_reference_items(items, valid_indices, idx_ent_map, matching_group, study_id, row, field_name):
        valid_pairs = []
        skipped = []
        seen_pairs = set()
        
        # 먼저 items를 쌍으로 구성
        pairs = []
        i = 0
        while i < len(items)-1:
            current_item = items[i].strip()
            next_item = items[i+1].strip()
            
            if next_item.startswith('idx'):
                pairs.append((current_item, next_item))
                i += 2
            else:
                i += 1
        
        # 각 쌍을 검증
        for entity, idx_str in pairs:
            entity_lower = entity.lower()  # 비교를 위한 lowercase 변환
            
            try:
                ref_idx = int(idx_str.split('idx')[-1])
                
                if ref_idx in valid_indices:
                    # 유효한 idx인 경우 - 원래 대소문자 유지
                    pair = (entity, f'idx{ref_idx}')
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        valid_pairs.append(pair)
                else:
                    # 유효하지 않은 idx인 경우, lowercase로 entity 매칭
                    matching_rows = matching_group[
                        (matching_group['ent'].str.lower().str.contains(entity_lower, regex=False)) & 
                        (matching_group['idx'].isin(valid_indices))
                    ]
                    
                    if not matching_rows.empty:
                        # 매칭되는 entity가 있는 경우
                        new_idx = int(matching_rows.iloc[0]['idx'])
                        # 원본 entity의 대소문자는 유지
                        pair = (entity, f'idx{new_idx}')
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            valid_pairs.append(pair)
                    else:
                        # 매칭되는 entity가 없는 경우
                        skipped.append((study_id, row['ent'], row[field_name], '매칭되는 entity 없음', entity))
                    
            except ValueError:
                skipped.append((study_id, row['ent'], row[field_name], f'잘못된 idx 형식: {idx_str}', entity))
        
        # 쌍으로 된 결과를 flatten하여 반환
        valid_items = [item for pair in valid_pairs for item in pair]
        return valid_items, skipped

    def process_row(row, valid_indices, idx_ent_map, matching_group, study_id):
        processed_row = row.copy()
        skipped_items = []

        for field in ['evidence', 'associate']:
            if pd.notna(row[field]) and row[field].strip():
                items = [item.strip() for item in row[field].split(',')]
                valid_items, skipped = process_reference_items(
                    items, valid_indices, idx_ent_map, matching_group, study_id, row, field
                )
                processed_row[field] = ', '.join(valid_items) if valid_items else None
                skipped_items.extend(skipped)

        return processed_row, skipped_items

    skipped_rows = []

    # 전체 데이터프레임 처리
    processed_df = report_level_df.copy()
    for study_id, group in report_level_df.groupby('study_id'):
        matching_group = matching_dataframe[matching_dataframe['study_id'] == study_id]
        valid_indices = set(group['idx'].dropna().astype(float).astype(int))
        
        idx_ent_map = {
            int(float(row['idx'])): (
                row['ent'].strip() if pd.notna(row['ent']) else '',
                row['location'].strip() if pd.notna(row['location']) else ''
            )
            for _, row in group.iterrows() 
            if pd.notna(row['idx'])
        }
        
        for idx in group.index:
            processed_row, skipped = process_row(
                processed_df.loc[idx],
                valid_indices,
                idx_ent_map,
                matching_group,
                study_id
            )
            processed_df.loc[idx] = processed_row
            skipped_rows.extend(skipped)

    # 결과 출력
    if skipped_rows:
        print("누락된 항목들:")
        for skipped in skipped_rows:
            print(f"Study ID: {skipped[0]}, Entity: {skipped[1]}, Original: {skipped[2]}, Reason: {skipped[3]}, Missing Entity: {skipped[4]}")

    return processed_df
