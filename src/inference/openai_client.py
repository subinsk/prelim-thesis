import os
import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


class OpenAIClient:
    """Client for OpenAI API inference."""

    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.request_count = 0

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        max_retries: int = 5
    ) -> str:
        """Generate response with rate limit handling."""

        self.request_count += 1
        if self.request_count % 100 == 0:
            print(f"  [Rate limit pause: {self.request_count} requests]")
            time.sleep(1)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                error_str = str(e)

                # Rate limit: wait and retry
                if "429" in error_str or "rate" in error_str.lower():
                    wait_time = 60 * (attempt + 1)
                    if wait_time > 600:
                        raise RuntimeError(
                            f"OpenAI rate limit exceeded after {attempt+1} retries. "
                            "Saving partial results..."
                        )
                    print(f"  Rate limited. Waiting {wait_time}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue

                # Other errors
                print(f"  OpenAI API Error: {e}")
                time.sleep(5 * (attempt + 1))

        raise RuntimeError(f"OpenAI: failed after {max_retries} retries")


if __name__ == "__main__":
    from prompt_templates import create_cot_prompt, extract_answer

    client = OpenAIClient()

    prompt = create_cot_prompt(
        question="What year was the director of Inception born?",
        doc1="Inception is a 2010 film directed by Christopher Nolan.",
        doc2="Christopher Nolan was born in 1970 in London."
    )

    response = client.generate(prompt)
    print(f"Response:\n{response}")
    print(f"\nExtracted answer: {extract_answer(response)}")
