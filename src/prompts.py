#TODO: Add we should make better prompts for each task

subjectivity_classification_prompt = """
Classify the following sentence as either 'subjective' or 'objective':

Sentence: {sentence}

Classification:
"""

subjectivity_uncertainty_score_prompt = """
Calculate the uncertainty score for the following sentence:

Sentence: {sentence}

Uncertainty Score:
"""

numeric_injection_prompt = """
Inject a numeric value into the following sentence without changing the contents of the sentence:

Sentence: {sentence}

Numeric Injection:
"""