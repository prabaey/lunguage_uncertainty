# Lunguage++

This repository provides resources for handling uncertainty in radiology reports, focusing on both explicit and implicit uncertainty. Our work addresses two key challenges in radiology report analysis:

1. **Explicit Uncertainty**: Uncertainty directly expressed through hedging phrases in reports
2. **Implicit Uncertainty**: Uncertainty arising from omitted findings and incomplete reasoning chains

![Overview of explicit and implicit uncertainty in chest x-ray reports](https://github.com/prabaey/lunguage_uncertainty/blob/main/figures/radiology_uncertainty.png)

## Overview

Radiology reports are crucial for clinical decision-making but often contain uncertainty that needs to be properly quantified and handled. This repository provides tools and resources to:

- Quantify explicit uncertainty through expert-validated hedging phrase rankings
- Model implicit uncertainty by expanding reports with systematic sub-findings
- Create uncertainty-aware structured representations of radiology reports

## Data Resources

Should you want to run the code in this repository, the `data_resources` folder should have the following structure: 
<pre>
data_resources/
├── evaluation_study/
│   ├── expert_participant_info.csv
│   └── expert_phrase_comparisons.csv
│   └── survey_uncertainty.pdf
├── hedging_phrase_comparisons/
│   └── comparisons_claude.jsonl
│   └── comparisons_gemini.jsonl   
│   └── comparisons_gpt.jsonl
│   └── comparisons_medgemma.jsonl
├── lunguage/
│   └── Lunguage.csv
│   └── Lunguage_vocab.csv
│   └── mimic-cxr-2.0.0-metadata.csv
├── ranking_log/
│   ├── llm_logs/
│   │   └── example.jsonl
│   ├── ranker_logs/
│   │   └── example.csv
│   └── rank_sentence_log.jsonl
├── Lunguage_plus_plus.csv
├── Lunguage_w_prob.csv
├── dx_pathway.csv
├── hedging_phrase_extracted.jsonl
├── hedging_phrase_vocab.jsonl
└── reference_ranking.csv
</pre>

The data in these folders can be downloaded from Physionet (**Our dataset is currently awaiting approval for publication by Physionet**).
The data in `data_resources/lunguage` can be downloaded from the Physionet project for Lunguage (**The Lunguage dataset is currently awaiting approval for publication by Physionet**).
The metadata file `mimic-cxr-2.0.0-metadata.csv` can be downloaded from [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr/2.0.0/).

What follows is a description of our dataset.

We publish Lunguage++ on Physionet, an expanded version of [Lunguage](https://arxiv.org/abs/2505.21190). The meaning of the relevant columns is the following. The columns that are added in Lunguage++ (by our explicit and implicit uncertainty pipelines) are shown in **bold**. 
- `idx`: index of finding-sentence pair in the report section
- `subject_id`: subject ID, links to patients in MIMIC-CXR
- `study_id`: study ID, links to chest x-ray studies in MIMIC-CXR
- `section`: report section this sentence was extracted from. In Lunguage++, we only consider findings (`find`) and impression (`impr`) sections.
- `sent`: one of the sentences in the report
- `ent`: finding that was extracted from the sentence, for which the remaining columns are structured attributes
- **`status`**: combination of `dx_status` and `dx_certainty` from Lunguage, indicates whether the finding (`ent`) is definitive positive (dp), definitive negative (dn), tentative positive (tp), or tentative negative (tn)
- **`prob`**: probability of the `ent` as expressed by the hedging phrases used in `sent`, quantified by our Explicit Uncertainty Framework.
- `location`, `evidence`, `associate`, `morphology`, `distribution`, `measurement`, `severity`, `comparison`, `onset`, `no change`, `improved`, `worsened`, `placement`, `past hx`, `other source`, `assessment limitations`: attributes of finding, see [Lunguage](https://arxiv.org/abs/2505.21190) for their meaning
- `report`: the full report this sentence was extracted from
- **`view_information`**: view of chest x-ray study and patient position, extracted from MIMIC-CXR meta-data, used to select the correct pathway in our Implicit Uncertainty Framework
- **`pathway`**: diagnostic pathway that is used to expand the disease with characteristic sub-findings in our Implicit Uncertainty Framework
- **`DxTP`**: whether the diagnosis/finding is current or in the past (if it is in the past, we should not expand it with key findings in our Implicit Uncertainty Framework)

We additionally publish the following resources:
- `hedging_phrase_extracted.jsonl`: hedging phrases extracted from every finding-sentence pair in Lunguage with a tentative label (`dx_certainty = tentative`)
- `hedging_phrase_vocab.jsonl`: vocabulary of 42 hedging phrases which occurred 10 or more times, each with an associated list of example sentences extracted from Lunguage
- `reference_ranking.csv`: reference ranking of 42 hedging phrases, constructed by applying the TrueSkill algorithm on LLM comparisons of pairs of phrases (found in `hedging_phrase_comparisons`)
- `dx_pathway.csv`: the expert-defined diagnostic pathways which can be used to expand diseases with their characteristic findings
- `ranking_log`, `hedging_phrase_comparisons`, and `evaluation_study`: additional data which is used in the demonstration notebook `run_explicit_uncertainty/build_explicit_uncertainty.ipynb`. 

## Explicit Uncertainty Framework

In this framework, we assign a probability to every finding-sentence pair in Lunguage, following the framework in the figure below. This results in a version of Lunguage with a new column called "prob", which stores this probability for every entry. This data can be accessed through `Lunguage_w_prob.csv` on Physionet and forms the starting point for the Implicit Uncertainty Framework (see next section). It is also contained in the final Lunguage++ dataset (`Lunguage_plus_plus.csv`) which can be found on Physionet.

![Overview of the explicit uncertainty framework: strategy for assigning probabilities to finding–sentence pairs with tentative certainty in the Lunguage dataset](https://github.com/prabaey/lunguage_uncertainty/blob/main/figures/explicit_uncertainty_overview.png)

To illustrate how these probabilities were obtained, we demonstrate every step of the pipeline in the notebook `run_explicit_uncertainty/build_explicit_uncertainty.ipynb`. 

## Implicit Uncertainty Framework

We handle implicit uncertainty through a diagnostic pathway expansion framework. This system systematically adds characteristic sub-findings derived from expert-defined diagnostic pathways for 14 common diagnoses.

![Overview of the Implicit Uncertainty Framework with Pathway Expansion](https://github.com/prabaey/lunguage_uncertainty/blob/main/figures/pathway_overview.png)

### Running the Pathway Expansion

To process reports using the diagnostic pathway expansion system:

1. Navigate to the `run_pathway` directory
2. Explore the diagnostic pathways and Lunguage++ using `run_pathway/exploration.ipynb`
3. Run the expansion script:
    1. Prepare required input files (via Physionet, **ur dataset is currently awaiting approval for publication by Physionet**):
        - `data_resources/dx_pathway.csv`: Diagnostic pathway definitions for 14 common diagnoses
        - `data_resources/lunguage/Lunguage_vocab.csv`:  Lunguage resource containing vocabulary and phrases (see [Lunguage paper](https://arxiv.org/abs/2505.21190))
        - `data_resources/Lunguage_w_prob.csv`: Lunguage resource with added 'prob' column for explicit uncertainty values
        - `data_resources/lunguage/mimic-cxr-2.0.0-metadata.csv`: MIMIC-CXR metadata file from [MIMIC-CXR-JPG](https://physionet.org/content/mimic-cxr/2.0.0/) for view information matching
    2. Install required packages by running:
        ```bash
        pip install -r requirements_for_pathway.txt
        ```

    3. Then execute the pathway expansion:
        ```bash
        # Basic usage
        python dx_pathway.py --resolve_conflicts --save_output

        # With optional arguments
        python dx_pathway.py \
            --vocab_path ../data_resources/lunguage/Lunguage_vocab.csv \
            --dx_pathway_path ../data_resources/dx_pathway.csv \
            --gold_dataset_path ../data_resources/Lunguage_w_prob.csv \
            --mimic_path ../data_resources/lunguage/mimic-cxr-2.0.0-metadata.csv \
            --output_dir ./output \
            --level report \  # Process at report level (integrates findings across sections) # Options: 'report' or 'section'
            --matching model \  # Use model-based similarity matching for entity/view/location # Options: 'model' or 'string'
            --resolve_conflicts \
            --save_output \
            --verbose
        ```
        
    The final Lunguage++ output will be saved as 'final_report_level_model.csv' in the output directory.
