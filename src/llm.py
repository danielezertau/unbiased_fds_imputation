from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

def load_env_file(dotenv_path="config/.env"):
    load_dotenv(dotenv_path)

def get_llm(model="gpt-4o"):
    return ChatOpenAI(
        model=model,
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )


def get_chain():
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        messages = [
            (
                "system",
                "You are a helpful assistant that detects functional dependencies in a database. Given a functional "
                "dependency that holds in an existing database, Say whether it is biased against some group in society. "
                "For example: sex -> capital.loss is biased, but occupation -> income and sex -> race is not."
                "Think about your answer, but eventually state BIASED or NOT-BIASED.",
            ),
            ("human", "{lhs} -> {rhs}"),
        ]
    )
    return prompt | llm

def is_fd_biased(lhs, rhs, retries=2):
    # If the LLM can't answer, return None as an indication of not knowing
    if retries == 0:
        return None
    chain = get_chain()
    llm_output = chain.invoke({
        "lhs": lhs,
        "rhs": rhs
    })

    if llm_output.additional_kwargs["refusal"] is True:
        return is_fd_biased(lhs, rhs, retries - 1)

    if "NOT-BIASED" in llm_output.content:
        return False
    if "BIASED" in llm_output.content:
        return True

    return is_fd_biased(lhs, rhs, retries - 1)
