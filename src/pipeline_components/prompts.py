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

First, analyze your reasoning step by step:
1. Examine the statement and identify key linguistic features (word choice, tone, factual vs. opinion-based content)
2. Consider why the proposed answer makes sense or doesn't make sense for this classification
3. Evaluate potential ambiguities or edge cases that might affect the classification
4. Reflect on how confident you are in the proposed answer based on your analysis

After your analysis, provide an uncertainty score (0-100).
IMPORTANT: Always provide your score like in the example below.

Example:
Statement: "I personally hate everything to do with the colour Red."
Proposed answer: subjective

Analysis:
1. The statement contains clear subjective indicators: "I personally" establishes first-person perspective, "hate" is an emotional verb expressing personal feeling, and the statement is about a personal preference rather than factual information.
2. The proposed answer "subjective" makes perfect sense because this sentence explicitly expresses the author's personal emotional stance and opinion about the color red, which directly aligns with the definition of subjective content.
3. There are no ambiguities here - the language is unambiguously personal and opinion-based with no factual claims that could be verified independently.
4. I am highly confident in this classification as the subjective markers are explicit and undeniable.

Uncertainty score: 95

Now analyze the given statement:
Statement: {sentence}
Proposed answer: {proposed_answer}

Analysis:
"""
