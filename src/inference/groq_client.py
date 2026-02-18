import os
import re
from groq import Groq
from dotenv import load_dotenv
import time

load_dotenv()


class GroqClient:
    """Client for Groq API inference."""

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = model
        self.request_count = 0

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        max_retries: int = 5
    ) -> str:
        """Generate response from LLM with smart rate limit handling."""

        self.request_count += 1
        if self.request_count % 50 == 0:
            print(f"  [Rate limit pause: {self.request_count} requests]")
            time.sleep(2)

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

                # Parse wait time from error message
                wait_match = re.search(r'Please try again in (\d+)m([\d.]+)s', error_str)
                if wait_match:
                    wait_mins = int(wait_match.group(1))
                    wait_secs = float(wait_match.group(2))
                    wait_time = wait_mins * 60 + wait_secs + 5  # add buffer

                    if wait_time > 600:  # >10 min = daily limit exhausted
                        raise RuntimeError(
                            f"Daily token limit reached. Retry in ~{wait_mins}m. "
                            "Saving partial results..."
                        )

                    print(f"  Rate limited. Waiting {wait_time:.0f}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue

                # Non-rate-limit error: short backoff
                print(f"  API Error: {e}")
                time.sleep(5 * (attempt + 1))

        raise RuntimeError(f"Failed after {max_retries} retries")


if __name__ == "__main__":
    from prompt_templates import create_cot_prompt, extract_answer

    client = GroqClient()

    prompt = create_cot_prompt(
        question="What year was the director of Inception born?",
        doc1="Inception is a 2010 film directed by Christopher Nolan.",
        doc2="Christopher Nolan was born in 1970 in London."
    )

    response = client.generate(prompt)
    print(f"Response:\n{response}")
    print(f"\nExtracted answer: {extract_answer(response)}")
