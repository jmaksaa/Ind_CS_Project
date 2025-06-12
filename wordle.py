import re
import random
import json
import wordlist
import openai
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- Configuration and Data (User will provide these later) ---

# Different prompting techniques (user will provide these exact strings)
# These are example templates. The '{feedback}' placeholder will be replaced with Wordle feedback.
PROMPT_TEMPLATES = {
    "basic": "Play Wordle. Your goal is to guess a 5-letter word. I will give you feedback. '🟩' means correct letter, correct position. '🟨' means correct letter, wrong position. '⬜' means wrong letter. Provide your guess on a new line prefixed with 'Guess: '. Current feedback: {feedback}",
    "chain_of_thought": "Let's play Wordle. Think step-by-step. First, analyze the feedback. Then, consider possible words. Then, check to make sure the word you guess is the best word, and if not, switch your word. Finally, make your guess. '🟩' means correct letter, correct position. '🟨' means correct letter, wrong position. '⬜' means wrong letter. Provide your guess on a new line prefixed with 'Guess: '. Current feedback: {feedback}",
    "persona": "You are the world champion in Wordle, and are in the grand finals for one million dollars. You know every strategy, every word, and are the best Wordle player in existence. Show the cheering crowds what you’ve got! '🟩' means correct letter, correct position. '🟨' means correct letter, wrong position. '⬜' means wrong letter. Provide your guess on a new line prefixed with 'Guess: '. Current feedback: {feedback}",
    "few_shot": "Example 1:\nInput:\nTurn 1: ADIEU\nFeedback: ⬜🟨⬜🟩⬜\nTurn 2: COUNT\nFeedback: ⬜🟩🟩🟩⬜\nOutput:\nThe Wordle answer is FOUNT. Guess: FOUNT\nExample 2:\nInput:\nTurn 1: STARE\nFeedback: 🟩🟨⬜⬜⬜\nTurn 2: SPACE\nFeedback: 🟩🟩⬜⬜⬜\nOutput:\nThe Wordle answer is not yet clear. A good next guess would be SALTY. Guess: SALTY\n\n🟩' means correct letter, correct position. '🟨' means correct letter, wrong position. '⬜' means wrong letter. Provide your guess on a new line prefixed with 'Guess: '. Current feedback: {feedback}"
}


MAX_GUESSES = 6
LLM_RETRY_ATTEMPTS = 2 # How many times to re-ask LLM for a valid guess
NUM_EXPERIMENT_RUNS = 25 # New: Number of times to run the entire experiment

# --- Helper Functions ---

def get_wordle_feedback(secret_word, guess):
    """
    Generates Wordle-style feedback for a given guess against the secret word.
    '🟩' (green): Correct letter and correct position.
    '🟨' (yellow): Correct letter but wrong position.
    '⬜' (gray): Letter not in the word.
    """
    feedback = ["⬜"] * 5
    secret_chars = list(secret_word)
    guess_chars = list(guess)

    # First pass: Check for green (correct letter, correct position)
    for i in range(5):
        if guess_chars[i] == secret_chars[i]:
            feedback[i] = "🟩"
            secret_chars[i] = None  # Mark as used
            guess_chars[i] = None   # Mark as used

    # Second pass: Check for yellow (correct letter, wrong position)
    for i in range(5):
        if guess_chars[i] is not None:
            if guess_chars[i] in secret_chars:
                feedback[i] = "🟨"
                # Mark the first occurrence of the character in secret_chars as used
                secret_chars[secret_chars.index(guess_chars[i])] = None

    return "".join(feedback)

def validate_guess(guess, valid_words):
    """
    Checks if a guess is a valid 5-letter word from the predefined list.
    """
    return len(guess) == 5 and guess.lower() in [w.lower() for w in valid_words]

def get_llm_guess(prompt_text, chat_history):
    """
    Sends a prompt to an LLM (ChatGPT) and gets a response.

    Args:
        prompt_text (str): The current prompt to send to the LLM.
        chat_history (list): A list of previous messages in the conversation,
                              formatted for the OpenAI Chat Completions API.

    Returns:
        str: The LLM's raw response content.
    """
    if client is None:
        print("OpenAI client not initialized. Cannot get LLM guess.")
        return "Error: LLM service unavailable."

    # print(f"--- LLM Prompt ---") # Suppressed for brevity during multiple runs
    # messages_for_api = chat_history + [{"role": "user", "content": prompt_text}]
    # print(f"Messages sent to LLM: {messages_for_api}")
    # print(f"--- End LLM Prompt ---")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=chat_history + [{"role": "user", "content": prompt_text}], # Pass history with current prompt
            temperature=0,
            max_tokens=1000
        )
        llm_response_content = response.choices[0].message.content
        return llm_response_content
    except openai.APIError as e:
        # print(f"OpenAI API Error: {e}") # Suppressed for brevity
        return f"Error: OpenAI API call failed: {e}"
    except Exception as e:
        # print(f"An unexpected error occurred during LLM call: {e}") # Suppressed for brevity
        return f"Error: An unexpected error occurred: {e}"

def extract_llm_guess(llm_response):
    """
    Extracts the 5-letter guess from the LLM's response using regex.
    Looks for "Guess: [five_letters]".
    """
    match = re.search(r"Guess:\s*([a-zA-Z]{5})", llm_response)
    if match:
        return match.group(1).lower()
    return None

# --- Main Game Logic ---

def play_wordle_game(secret_word, prompt_type, prompt_template, valid_words):
    """
    Plays a single Wordle game with the LLM using a specific prompt type.

    Args:
        secret_word (str): The Wordle word to guess.
        prompt_type (str): The name of the prompting technique (e.g., "basic").
        prompt_template (str): The template string for the prompt.
        valid_words (list): List of valid Wordle words.

    Returns:
        dict: Results of the game (guesses, outcome, etc.).
    """
    # print(f"\n--- Starting Wordle Game with Prompt Type: {prompt_type} ---") # Suppressed for brevity
    # print(f"Secret Word: {secret_word}") # Suppressed for brevity

    chat_history = []
    guesses = []
    feedbacks = [] # To store all past feedbacks

    for guess_num in range(1, MAX_GUESSES + 1):
        prompt_text = ""
        if guess_num == 1:
            # First turn: Use the full template
            prompt_text = prompt_template.format(feedback="Start")
        else:
            # Subsequent turns: Provide all previous guesses and feedback, then the current feedback
            history_str = "\n".join([f"Turn {i+1}: {guesses[i].upper()} -> {feedbacks[i]}" for i in range(len(guesses))])
            
            # Construct the prompt based on the prompt type for subsequent turns
            if prompt_type == "few_shot":
                # For few-shot, just append the new feedback after previous examples
                prompt_text = f"{prompt_template.split('Current feedback:')[0].strip()}\nCurrent feedback: {feedbacks[-1]}"
            elif prompt_type == "chain_of_thought":
                prompt_text = f"Continue playing Wordle. Think step-by-step. Current game history:\n{history_str}\nNew feedback: {feedbacks[-1]}\nProvide your guess on a new line prefixed with 'Guess: '."
            elif prompt_type == "persona":
                prompt_text = f"You are still the world champion. Current game history:\n{history_str}\nNew feedback: {feedbacks[-1]}\nShow the cheering crowds what you’ve got! Provide your guess on a new line prefixed with 'Guess: '."
            else: # Basic prompt or any other type
                prompt_text = f"Current game history:\n{history_str}\nNew feedback: {feedbacks[-1]}\nProvide your guess on a new line prefixed with 'Guess: '."


        llm_guess = None
        raw_llm_response = ""
        
        for attempt in range(LLM_RETRY_ATTEMPTS):
            raw_llm_response = get_llm_guess(prompt_text, chat_history)
            llm_guess = extract_llm_guess(raw_llm_response)

            chat_history.append({"role": "user", "content": prompt_text})
            chat_history.append({"role": "assistant", "content": raw_llm_response})

            if llm_guess and validate_guess(llm_guess, valid_words):
                # print(f"LLM Guess ({guess_num}/{MAX_GUESSES}): {llm_guess}") # Suppressed for brevity
                break
            else:
                # print(f"LLM provided an invalid guess. Raw LLM response: '{raw_llm_response}' (Attempt {attempt + 1}/{LLM_RETRY_ATTEMPTS}). Re-asking...") # Suppressed for brevity
                prompt_text = "Invalid guess. Please provide a valid 5-letter English word from the Wordle dictionary. Remember to prefix your guess with 'Guess: '."
                if attempt == LLM_RETRY_ATTEMPTS - 1:
                    # print(f"LLM failed to provide a valid guess after {LLM_RETRY_ATTEMPTS} attempts. Aborting game.") # Suppressed for brevity
                    return {
                        "prompt_type": prompt_type,
                        "secret_word": secret_word,
                        "guesses": guesses,
                        "outcome": "aborted_invalid_guess",
                        "num_guesses": len(guesses),
                        "chat_history": chat_history
                    }

        if not llm_guess:
            # print("Failed to get a valid guess from LLM.") # Suppressed for brevity
            return {
                "prompt_type": prompt_type,
                "secret_word": secret_word,
                "guesses": guesses,
                "outcome": "failed_to_get_guess",
                "num_guesses": len(guesses),
                "chat_history": chat_history
            }

        guesses.append(llm_guess)
        current_feedback = get_wordle_feedback(secret_word, llm_guess)
        feedbacks.append(current_feedback) # Store current feedback
        # print(f"Feedback: {current_feedback}") # Suppressed for brevity

        if llm_guess == secret_word:
            # print(f"LLM guessed the word '{secret_word}' in {guess_num} guesses!") # Suppressed for brevity
            return {
                "prompt_type": prompt_type,
                "secret_word": secret_word,
                "guesses": guesses,
                "outcome": "solved",
                "num_guesses": guess_num,
                "chat_history": chat_history
            }

    # print(f"LLM did not guess the word '{secret_word}' within {MAX_GUESSES} guesses.") # Suppressed for brevity
    return {
        "prompt_type": prompt_type,
        "secret_word": secret_word,
        "guesses": guesses,
        "outcome": "failed",
        "num_guesses": MAX_GUESSES,
        "chat_history": chat_history
    }

def run_experiment(secret_word, prompt_templates, valid_words):
    """
    Runs the Wordle game for each prompting technique and collects results.

    Args:
        secret_word (str): The Wordle word to guess.
        prompt_templates (dict): Dictionary of prompt types and their templates.
        valid_words (list): List of valid Wordle words.

    Returns:
        list: A list of game result dictionaries.
    """
    all_results_for_this_word = []
    # print(f"\n--- Running Experiment for Secret Word: {secret_word} ---") # Suppressed for brevity
    for prompt_type, template in prompt_templates.items():
        game_result = play_wordle_game(secret_word, prompt_type, template, valid_words)
        all_results_for_this_word.append(game_result)
        # print("-" * 50) # Separator for readability # Suppressed for brevity
    return all_results_for_this_word

def analyze_results(all_experiment_results, NUM_EXPERIMENT_RUNS):
    """
    Analyzes the collected game results from multiple runs to determine the best prompting technique
    and saves the analysis to 'analyzed_results.json'.
    """
    summary = {}

    for run_results in all_experiment_results:
        for res in run_results:
            p_type = res["prompt_type"]
            outcome = res["outcome"]
            num_guesses = res["num_guesses"]

            if p_type not in summary:
                summary[p_type] = {"solved": 0, "failed": 0, "aborted_invalid_guess": 0, "total_guesses_on_solve": 0, "games_played": 0}

            summary[p_type]["games_played"] += 1
            if outcome == "solved":
                summary[p_type]["solved"] += 1
                summary[p_type]["total_guesses_on_solve"] += num_guesses
            elif outcome == "failed":
                summary[p_type]["failed"] += 1
            elif outcome == "aborted_invalid_guess":
                summary[p_type]["aborted_invalid_guess"] += 1

    # Prepare data for JSON output
    analysis_output = {"summary_by_prompt_type": {}}
    for p_type, data in summary.items():
        avg_guesses = (data["total_guesses_on_solve"] / data["solved"]) if data["solved"] > 0 else "N/A"
        solve_rate = (data["solved"] / data["games_played"]) * 100 if data["games_played"] > 0 else 0
        analysis_output["summary_by_prompt_type"][p_type] = {
            "games_played": data['games_played'],
            "solved": data['solved'],
            "solve_rate": f"{solve_rate:.2f}%",
            "failed": data['failed'],
            "aborted_invalid_guess": data['aborted_invalid_guess'],
            "average_guesses_on_solve": avg_guesses
        }

    # Determine the best performing prompt (highest solve rate, then lowest average guesses)
    best_prompt = None
    max_solve_rate = -1
    min_avg_guesses = float('inf')
    best_prompt_details = {}

    for p_type, data in summary.items():
        if data["games_played"] > 0:
            solve_rate = data["solved"] / data["games_played"]
            current_avg_guesses = (data["total_guesses_on_solve"] / data["solved"]) if data["solved"] > 0 else float('inf')

            if solve_rate > max_solve_rate:
                max_solve_rate = solve_rate
                min_avg_guesses = current_avg_guesses
                best_prompt = p_type
                best_prompt_details = {
                    "prompt_type": p_type,
                    "solve_rate": f"{solve_rate*100:.2f}%",
                    "average_guesses_on_solve": current_avg_guesses
                }
            elif solve_rate == max_solve_rate:
                if current_avg_guesses < min_avg_guesses:
                    min_avg_guesses = current_avg_guesses
                    best_prompt = p_type
                    best_prompt_details = {
                        "prompt_type": p_type,
                        "solve_rate": f"{solve_rate*100:.2f}%",
                        "average_guesses_on_solve": current_avg_guesses
                    }

    if best_prompt:
        analysis_output["best_performing_prompt"] = best_prompt_details
        analysis_output["best_performing_prompt"]["analysis_based_on_runs"] = NUM_EXPERIMENT_RUNS
    else:
        analysis_output["best_performing_prompt"] = "Could not determine a best performing prompt type."

    # Dump the analysis to a JSON file
    with open("analyzed_results.json", "w") as f:
        json.dump(analysis_output, f, indent=4)

    print("Analysis complete. Results saved to 'analyzed_results.json'.")

# --- Main Execution ---
if __name__ == "__main__":
    all_experiment_results_across_runs = []
    
    print(f"--- Running {NUM_EXPERIMENT_RUNS} Wordle experiments ---")

    for i in range(NUM_EXPERIMENT_RUNS):
        # Choose a random secret word for each run
        secret_word_for_run = random.choice(wordlist.VALID_WORDS).lower()
        print(f"\n--- Experiment Run {i+1}/{NUM_EXPERIMENT_RUNS} with Secret Word: '{secret_word_for_run}' ---")
        
        # Run the experiment for the current secret word across all prompt types
        run_results = run_experiment(secret_word_for_run, PROMPT_TEMPLATES, wordlist.VALID_WORDS)
        all_experiment_results_across_runs.append(run_results)
        print(f"--- Finished Run {i+1} ---")

    # Analyze and display aggregated results
    analyze_results(all_experiment_results_across_runs,NUM_EXPERIMENT_RUNS)

    # Dump all results to results.json
    with open("results.json", "w") as f:
        json.dump(all_experiment_results_across_runs, f, indent=2)
    print(f"\nDetailed results from {NUM_EXPERIMENT_RUNS} runs have been dumped to results.json")