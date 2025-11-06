VALID_ATTR_CATEGORY2 = [
    'subject_id', 'study_id', 'sequence', 'section', 'report', 'annotator', 'associate', 
    'view_information', 'placement', 'other source', 'assessment limitations', 'past hx', 
    'morphology', 'distribution', 'measurement', 'severity', 'comparison', 'onset', 
    'improved', 'no change', 'worsened'
]

REPORT_LEVEL_CATEGORY = VALID_ATTR_CATEGORY2 + ['DxTP']

VALID_ATTR_CATEGORY = [
    'morphology', 'distribution', 'measurement', 'severity', 
    'onset', 'improved', 'no change', 'worsened'
]

CLINICAL_ATTR_CAT = [
    'placement', 'other source', 'assessment limitations', 'past hx', 'morphology', 
    'distribution', 'measurement', 'severity', 'comparison', 'onset', 
    'improved', 'no change', 'worsened'
]

ATTRIBUTE_NOT_ALLOWED_SUBCATEGORY = [
    'Medical Devices', 'Symptoms and Signs', 'Treatment and Medications', 'Procedures and Surgeries'
]

ATTRIBUTE_NOT_ALLOWED_CATEGORY = ['ncd', 'cof', 'patient info.', 'oth']
