#TODO: Add we should make better prompts for each task

subjectivity_classification_prompt = """
Rate the subjectivity of the following statement: "{sentence}"

Your task is to classify this statement as either objective or subjective.

Objective statements present factual information that can be verified 
independently through evidence, data, or observation. They describe measurable 
events, established facts, or verifiable claims without personal interpretation 
or bias.

Subjective statements express personal opinions, emotions, beliefs, interpretations,
or value judgments. They reflect individual perspectives, preferences, or evaluations 
that cannot be independently verified as universally true or false.

Answer with either "objective" or "subjective".
"""

subjectivity_uncertainty_score_prompt = """
Statement: {sentence}
Proposed answer: {proposed_answer}

Your task is to rate the uncertainty of the proposed subjectivity classification on a score 
from 0 to 100, where 0 represents definitely uncertain and 100 represents definitely certain.

Objective statements present factual information that can be verified 
independently through evidence, data, or observation. They describe measurable 
events, established facts, or verifiable claims without personal interpretation 
or bias.

Subjective statements express personal opinions, emotions, beliefs, interpretations,
or value judgments. They reflect individual perspectives, preferences, or evaluations 
that cannot be independently verified as universally true or false.

Please, only answer with your score.
"""

numeric_injection_prompt = """
Inject a numeric value into the following sentence without changing the contents of the sentence:

Sentence: {sentence}

Numeric Injection:
"""