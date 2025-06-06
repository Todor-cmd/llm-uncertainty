subjectivity_classification_prompt = """
You are an expert in subjectivity classification. Rate the subjectivity of the following sentence from a news article on a scale from 0 to 100, where:
- 0 represents completely objective (presents factual information without personal perspective)
- 100 represents completely subjective (expresses the author's personal views, opinions, or bias)

IMPORTANT: ONLY respond with the numeric score (0-100) like in the following example:

Example:
Sentence: "I personally hate eveything to do with the colour Red."
Answer: 100

Sentence: "{sentence}"
Answer:"""

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