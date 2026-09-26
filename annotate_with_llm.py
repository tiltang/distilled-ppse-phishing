"""Step 1 of 3 -- the teacher: an LLM scores every email on the five PPSE principles.

Llama 3.1 70B reads one email and one principle at a time. It returns a score from 0 to 1
(how strongly the principle is present in the email), the passages that support the score,
and its reasoning. This runs once, offline. The scores become the training targets of step 2.

This is reference code. It shows the logic of the step; it is not a ready-to-run job.
`call_llm` has to be connected to wherever the model is served.
"""
import pandas as pd
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

LLM_NAME = "meta-llama/Llama-3.1-70B-Instruct"
TEMPERATURE = 0.0
SEED = 0

# The principle definitions of Ferreira et al. (2015).
PRINCIPLES = {
    "authority": (
        "Authority",
        "Society trains people not to question authority so they are conditioned to respond "
        "to it. People usually follow an expert or pretense of authority and do a great deal "
        "for someone they think is an authority."),
    "social_proof": (
        "Social Proof",
        "People tend to mimic what the majority of people do or seem to be doing. People let "
        "their guard and suspicion down when everyone else appears to share the same "
        "behaviours and risks. In this way, they will not be held solely responsible for "
        "their actions."),
    "liking_similarity_deception": (
        "Liking, Similarity & Deception",
        "People prefer to abide to whom (they think) they know or like, or to whom they are "
        "similar to or familiar with, as well as attracted to."),
    "commitment_reciprocation_consistency": (
        "Commitment, Reciprocation & Consistency",
        "People feel more confident in their decision once they commit (publically) to a "
        "specific action and need to follow it through until the end. This is true whether "
        "in the workplace, or in a situation when their action is illegal. People have "
        "tendency to believe what others say and need, and they want to appear consistent "
        "in what they do, for instance, when they owe a favour. There is an automatic "
        "response of repaying a favour."),
    "distraction": (
        "Distraction",
        "People focus on one thing and ignore other things that may happen without them "
        "noticing; they focus attention on what they can gain, what they need, what they can "
        "lose or miss out on, or if that thing will soon be unavailable, has been censored, "
        "restricted or will be more expensive later. These distractions can heighten "
        "people's emotional state and make them forget other logical facts to consider when "
        "making decisions."),
}

PROMPT = """You are a cybersecurity analyst specializing in social engineering attacks.

Persuasion principles in social engineering are psychological techniques that attackers
use to manipulate a victim into performing an action or revealing information.

Your role is to detect the intensity of the presence of {ppse_name} in the email,
on a scale from 0 to 1, according to its formal definition.

Respond only in JSON, following this output model:
{format_instructions}

{ppse_name} definition: {ppse_def}
Email: {email}

Begin!"""


class PPSEAnnotation(BaseModel):
    """The output model. Its JSON schema is what fills {format_instructions}."""
    intensity: float = Field(ge=0, le=1, description="How strongly the principle is present, 0 to 1")
    evidence: list[str] = Field(description="Passages quoted from the email that support the score")
    reasoning: str = Field(description="Step-by-step explanation of the score")


parser = PydanticOutputParser(pydantic_object=PPSEAnnotation)


def call_llm(prompt: str) -> str:
    """Send one prompt to LLM_NAME with TEMPERATURE and SEED; return the raw text reply."""
    raise NotImplementedError("connect this to the server that hosts the model")


def annotate_email(email: str) -> dict:
    """Five separate calls, one per principle, so one principle cannot colour another's score."""
    scores = {}
    for key, (name, definition) in PRINCIPLES.items():
        prompt = PROMPT.format(ppse_name=name, ppse_def=definition, email=email,
                               format_instructions=parser.get_format_instructions())
        scores[key] = parser.parse(call_llm(prompt)).intensity
    return scores


def annotate_corpus(df: pd.DataFrame, text_col: str = "email_text") -> pd.DataFrame:
    """The corpus with five new columns, one score per principle."""
    scores = pd.DataFrame([annotate_email(t) for t in df[text_col].astype(str)], index=df.index)
    return df.join(scores)


if __name__ == "__main__":
    emails = pd.read_csv("emails.csv")                  # email_id, email_text, source, label, split
    annotate_corpus(emails).to_csv("emails_annotated.csv", index=False)
