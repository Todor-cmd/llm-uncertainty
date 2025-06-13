subjectivity_classification_prompt = """
Rate the subjectivity of the following sentence from a news article: "{sentence}"

Your task is to rate this sentence on a scale from 0 to 100, where:
- 0 represents completely objective (presents an objective view on the topic without the author's personal perspective)
- 100 represents completely subjective (expresses the subjective view of the author behind it)

Objective sentences present factual information, events, or statements without the author's personal interpretation, opinion, or bias.

Subjective sentences express the author's personal views, opinions, interpretations, evaluations, or emotional stance on the topic.

Please respond with only the numeric score (0-100).
"""

subjectivity_classification_binary_prompt = """
Your task is to classify sentence as either subjective or objective.
- If the sentence expresses the subjective view of the author, classify it as "subjective".
- If the sentence presents an objective view on the topic, classify it as "objective".
Please respond with only the word "subjective" or "objective".

Example:
Sentence: "The new policy is a disaster for the economy."
Response: "subjective"
Sentence: "The new policy was implemented on January 1st."
Response: "objective"

For the following sentence, please provide your classification:
Sentence: {sentence}
Response: 
"""

subjectivity_uncertainty_score_prompt = """
Statement: {sentence}
Proposed answer: {proposed_answer}

Your task is to rate the uncertainty of the proposed subjectivity classification on a score 
from 0 to 100, where 0 represents completley uncertain and 100 represents completley certain.

This sentence is from a news article. The classification task is to distinguish whether it 
expresses the subjective view of the author behind it or presents an objective view on the 
covered topic instead.

Objective sentences present factual information, events, or statements without the author's 
personal interpretation, opinion, or bias.

Subjective sentences express the author's personal views, opinions, interpretations, 
evaluations, or emotional stance on the topic.

Please, only answer with your score.
"""

numeric_injection_prompt = """
Inject a numeric value into the following sentence without changing the contents of the sentence:

Sentence: {sentence}

Numeric Injection:
"""