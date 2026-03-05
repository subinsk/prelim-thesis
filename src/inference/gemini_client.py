import os
import time
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class GeminiClient:
    """Client for Google Gemini API inference."""

    def __init__(self, model: str = "gemini-2.5-flash-lite"):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model
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

        # Gemini free tier: 15 req/min — pause every 14 requests
        if self.request_count % 14 == 0:
            print(f"  [Rate limit pause: {self.request_count} requests]")
            time.sleep(4)

        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                if response.text:
                    return response.text.strip()
                return ""

            except Exception as e:
                error_str = str(e)

                # Rate limit: wait and retry
                if "429" in error_str or "quota" in error_str.lower() or "rate" in error_str.lower():
                    wait_time = 60 * (attempt + 1)
                    if wait_time > 600:
                        raise RuntimeError(
                            f"Gemini rate limit exceeded after {attempt+1} retries. "
                            "Saving partial results..."
                        )
                    print(f"  Rate limited. Waiting {wait_time}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue

                # Other errors
                print(f"  Gemini API Error: {e}")
                time.sleep(5 * (attempt + 1))

        raise RuntimeError(f"Gemini: failed after {max_retries} retries")


if __name__ == "__main__":
    from prompt_templates import create_cot_prompt, extract_answer

    client = GeminiClient()

    prompt = create_cot_prompt(
        question="What year was the director of Inception born?",
        doc1="Inception is a 2010 film directed by Christopher Nolan.",
        doc2="Christopher Nolan was born in 1970 in London."
    )

    response = client.generate(prompt)
    print(f"Response:\n{response}")
    print(f"\nExtracted answer: {extract_answer(response)}")
