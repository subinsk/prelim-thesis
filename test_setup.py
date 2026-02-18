import os
import sys
import json

def main():
    print("=" * 50)
    print("THESIS PROJECT SETUP CHECK")
    print("=" * 50)

    # Step 1: Check .env
    print("\n[1/4] Checking .env file...")
    if not os.path.exists('.env'):
        with open('.env', 'w') as f:
            f.write("GROQ_API_KEY=your_groq_api_key_here\n")
        print("  Created .env file - ADD YOUR GROQ API KEY!")
        print("  Edit .env and replace 'your_groq_api_key_here' with your key")
        print("  Get a key at: https://console.groq.com/")
        sys.exit(1)

    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        print("  .env exists but GROQ_API_KEY is not set!")
        print("  Edit .env and add your Groq API key")
        print("  Get a key at: https://console.groq.com/")
        sys.exit(1)
    print("  .env found with API key set")

    # Step 2: Test Groq API
    print("\n[2/4] Testing Groq API connection...")
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Say 'API working!' in exactly 2 words."}],
            max_tokens=10
        )
        reply = response.choices[0].message.content.strip()
        print(f"  Groq API response: {reply}")
    except Exception as e:
        print(f"  Groq API Error: {e}")
        print("  Check your API key and internet connection")
        sys.exit(1)

    # Step 3: Download HotpotQA
    print("\n[3/4] Checking HotpotQA dataset...")
    os.makedirs('data/hotpotqa', exist_ok=True)

    if not os.path.exists('data/hotpotqa/dev.json'):
        print("  Downloading HotpotQA dev set (this may take a minute)...")
        import requests
        url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
        response = requests.get(url)
        with open('data/hotpotqa/dev.json', 'w') as f:
            json.dump(response.json(), f)
        print("  Downloaded HotpotQA!")
    else:
        print("  HotpotQA already exists")

    # Step 4: Verify dataset
    print("\n[4/4] Verifying dataset...")
    with open('data/hotpotqa/dev.json', 'r') as f:
        data = json.load(f)

    total = len(data)
    bridge = [ex for ex in data if ex['type'] == 'bridge']
    print(f"  Total examples: {total}")
    print(f"  Bridge questions: {len(bridge)}")
    print(f"  Sample question: {bridge[0]['question']}")
    print(f"  Sample answer: {bridge[0]['answer']}")

    print("\n" + "=" * 50)
    print("Setup complete! Ready to run experiments.")
    print("=" * 50)
    print("\nNext step:")
    print("  python experiments/run_conflict_experiment.py")


if __name__ == "__main__":
    main()
