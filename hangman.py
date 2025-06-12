import re
import random
import json
import wordlist # Assuming wordlist.VALID_WORDS exists and contains suitable words
import openai
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# --- Configuration and Data ---

# Different prompting techniques for Hangman
PROMPT_TEMPLATES = {
    "basic": "Play Hangman. Your goal is to guess letters to reveal a hidden word. You get {max_incorrect_guesses} incorrect guesses. If you guess a letter, I will show you the updated word (e.g., H_LLO). If you guess incorrectly, your incorrect guess count goes up. Already guessed letters: {guessed_letters}. Current word state: {display_word}. Provide your single letter guess on a new line prefixed with 'Guess: '.",
    "chain_of_thought": "Let's play Hangman. Think step-by-step. First, analyze the current word state and previously guessed letters. Then, consider common letters or letters that would reveal significant parts of the word. Finally, make your single letter guess. You get {max_incorrect_guesses} incorrect guesses. Already guessed letters: {guessed_letters}. Current word state: {display_word}. Provide your single letter guess on a new line prefixed with 'Guess: '.",
    "persona": "You are the world champion in Hangman, a master of deduction and vocabulary. The crowd is on the edge of their seats! You get {max_incorrect_guesses} incorrect guesses. Show them how it's done! Already guessed letters: {guessed_letters}. Current word state: {display_word}. Provide your single letter guess on a new line prefixed with 'Guess: '.",
    "few_shot": "Example 1:\nInput:\nSecret Word Length: 5\nIncorrect Guesses Left: 6\nGuessed Letters: []\nCurrent Word State: _ _ _ _ _\nOutput:\nI will start with a common letter. Guess: E\n\nExample 2:\nInput:\nSecret Word Length: 5\nIncorrect Guesses Left: 5\nGuessed Letters: ['e']\nCurrent Word State: _ E _ _ _\nOutput:\nNow that I know 'E' is in the word, I will try 'A'. Guess: A\n\nExample 3:\nInput:\nSecret Word Length: 6\nIncorrect Guesses Left: 3\nGuessed Letters: ['e', 'a', 't', 's']\nCurrent Word State: _ E A T _ S\nOutput:\nGiven the pattern, 'U' seems like a good fit. Guess: U\n\nPlay Hangman. You get {max_incorrect_guesses} incorrect guesses. Already guessed letters: {guessed_letters}. Current word state: {display_word}. Provide your single letter guess on a new line prefixed with 'Guess: '."
}


MAX_INCORRECT_GUESSES = 7 # Standard Hangman has 6-7 incorrect guesses
LLM_RETRY_ATTEMPTS = 2
NUM_EXPERIMENT_RUNS = 25

# --- Helper Functions ---

def get_hangman_feedback(secret_word, guess, display_word, incorrect_guesses_count):
    """
    Updates the display word and incorrect guess count based on the letter guess.
    """
    guess = guess.lower()
    secret_word_lower = secret_word.lower()
    
    updated_display_word_list = list(display_word.replace(" ", "")) # Convert 'H _ L L O' to ['H', '_', 'L', 'L', 'O']

    correct_guess = False
    for i, char in enumerate(secret_word_lower):
        if char == guess:
            updated_display_word_list[i] = guess
            correct_guess = True
    
    if not correct_guess:
        incorrect_guesses_count += 1
    
    return " ".join(updated_display_word_list), incorrect_guesses_count, correct_guess

def validate_guess_hangman(guess, guessed_letters):
    """
    Checks if a guess is a single, alphabetic letter that hasn't been guessed before.
    """
    return len(guess) == 1 and guess.isalpha() and guess.lower() not in [l.lower() for l in guessed_letters]

def get_llm_guess(prompt_text, chat_history):
    """
    Sends a prompt to an LLM (ChatGPT) and gets a response.
    """
    if client is None:
        print("OpenAI client not initialized. Cannot get LLM guess.")
        return "Error: LLM service unavailable."

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=chat_history + [{"role": "user", "content": prompt_text}],
            temperature=0,
            max_tokens=1000
        )
        llm_response_content = response.choices[0].message.content
        return llm_response_content
    except openai.APIError as e:
        return f"Error: OpenAI API call failed: {e}"
    except Exception as e:
        return f"Error: An unexpected error occurred: {e}"

def extract_llm_guess(llm_response):
    """
    Extracts the single letter guess from the LLM's response using regex.
    Looks for "Guess: [single_letter]".
    """
    match = re.search(r"Guess:\s*([a-zA-Z]{1})", llm_response)
    if match:
        return match.group(1).lower()
    return None

# --- Main Game Logic ---

def play_hangman_game(secret_word, prompt_type, prompt_template, valid_words):
    """
    Plays a single Hangman game with the LLM using a specific prompt type.
    """
    chat_history = []
    guessed_letters = []
    incorrect_guesses_count = 0
    
    # Initialize display_word with underscores
    display_word = "_ " * len(secret_word)
    display_word = display_word.strip() # Remove trailing space

    while "_" in display_word and incorrect_guesses_count < MAX_INCORRECT_GUESSES:
        prompt_text = ""
        
        # Prepare the prompt with current game state
        formatted_guessed_letters = ', '.join(f"'{l}'" for l in sorted(list(set(guessed_letters))))
        
        if prompt_type == "few_shot":
            # Few-shot prompts have a fixed structure at the beginning, then dynamically added current state
            base_prompt = prompt_template.split('Play Hangman.')[0].strip()
            prompt_text = f"{base_prompt}\n\nPlay Hangman. You get {MAX_INCORRECT_GUESSES} incorrect guesses. Already guessed letters: [{formatted_guessed_letters}]. Current word state: {display_word}. Provide your single letter guess on a new line prefixed with 'Guess: '."
        else:
            prompt_text = prompt_template.format(
                max_incorrect_guesses=MAX_INCORRECT_GUESSES - incorrect_guesses_count,
                guessed_letters=f"[{formatted_guessed_letters}]",
                display_word=display_word
            )

        llm_guess = None
        raw_llm_response = ""
        
        for attempt in range(LLM_RETRY_ATTEMPTS):
            raw_llm_response = get_llm_guess(prompt_text, chat_history)
            llm_guess = extract_llm_guess(raw_llm_response)

            chat_history.append({"role": "user", "content": prompt_text})
            chat_history.append({"role": "assistant", "content": raw_llm_response})

            if llm_guess and validate_guess_hangman(llm_guess, guessed_letters):
                break
            else:
                prompt_text = f"Invalid guess. Please provide a single, un-guessed alphabetic letter. Remember to prefix your guess with 'Guess: '. You have already guessed: [{formatted_guessed_letters}]."
                if attempt == LLM_RETRY_ATTEMPTS - 1:
                    return {
                        "prompt_type": prompt_type,
                        "secret_word": secret_word,
                        "guessed_letters": guessed_letters,
                        "outcome": "aborted_invalid_guess",
                        "incorrect_guesses": incorrect_guesses_count,
                        "chat_history": chat_history
                    }

        if not llm_guess:
            return {
                "prompt_type": prompt_type,
                "secret_word": secret_word,
                "guessed_letters": guessed_letters,
                "outcome": "failed_to_get_guess",
                "incorrect_guesses": incorrect_guesses_count,
                "chat_history": chat_history
            }

        guessed_letters.append(llm_guess)
        
        new_display_word, new_incorrect_guesses_count, correct_guess = \
            get_hangman_feedback(secret_word, llm_guess, display_word, incorrect_guesses_count)
        
        display_word = new_display_word
        incorrect_guesses_count = new_incorrect_guesses_count

        # Provide feedback to the LLM for the next turn
        if correct_guess:
            feedback_message = f"Correct! The word is now: {display_word}. You have {MAX_INCORRECT_GUESSES - incorrect_guesses_count} incorrect guesses left."
        else:
            feedback_message = f"Incorrect. That letter is not in the word. You now have {MAX_INCORRECT_GUESSES - incorrect_guesses_count} incorrect guesses left. Current word state: {display_word}."
        
        # This feedback isn't directly used in the prompt_text construction above, but added to chat_history
        # to provide context for the LLM.
        chat_history.append({"role": "system", "content": feedback_message})


    if "_" not in display_word:
        return {
            "prompt_type": prompt_type,
            "secret_word": secret_word,
            "guessed_letters": guessed_letters,
            "outcome": "solved",
            "incorrect_guesses": incorrect_guesses_count,
            "chat_history": chat_history
        }
    else:
        return {
            "prompt_type": prompt_type,
            "secret_word": secret_word,
            "guessed_letters": guessed_letters,
            "outcome": "failed",
            "incorrect_guesses": incorrect_guesses_count,
            "chat_history": chat_history
        }

def run_experiment(secret_word, prompt_templates, valid_words):
    """
    Runs the Hangman game for each prompting technique and collects results.
    """
    all_results_for_this_word = []
    for prompt_type, template in prompt_templates.items():
        game_result = play_hangman_game(secret_word, prompt_type, template, valid_words)
        all_results_for_this_word.append(game_result)
    return all_results_for_this_word

def analyze_results(all_experiment_results, NUM_EXPERIMENT_RUNS):
    """
    Analyzes the collected game results from multiple runs to determine the best prompting technique
    and saves the analysis to 'analyzed_results_hangman.json'.
    """
    summary = {}

    for run_results in all_experiment_results:
        for res in run_results:
            p_type = res["prompt_type"]
            outcome = res["outcome"]
            incorrect_guesses = res["incorrect_guesses"]

            if p_type not in summary:
                summary[p_type] = {"solved": 0, "failed": 0, "aborted_invalid_guess": 0, "total_incorrect_guesses_on_solve": 0, "games_played": 0}

            summary[p_type]["games_played"] += 1
            if outcome == "solved":
                summary[p_type]["solved"] += 1
                summary[p_type]["total_incorrect_guesses_on_solve"] += incorrect_guesses
            elif outcome == "failed":
                summary[p_type]["failed"] += 1
            elif outcome == "aborted_invalid_guess":
                summary[p_type]["aborted_invalid_guess"] += 1

    # Prepare data for JSON output
    analysis_output = {"summary_by_prompt_type": {}}
    for p_type, data in summary.items():
        avg_incorrect_guesses = (data["total_incorrect_guesses_on_solve"] / data["solved"]) if data["solved"] > 0 else "N/A"
        solve_rate = (data["solved"] / data["games_played"]) * 100 if data["games_played"] > 0 else 0
        analysis_output["summary_by_prompt_type"][p_type] = {
            "games_played": data['games_played'],
            "solved": data['solved'],
            "solve_rate": f"{solve_rate:.2f}%",
            "failed": data['failed'],
            "aborted_invalid_guess": data['aborted_invalid_guess'],
            "average_incorrect_guesses_on_solve": avg_incorrect_guesses
        }

    # Determine the best performing prompt (highest solve rate, then lowest average incorrect guesses)
    best_prompt = None
    max_solve_rate = -1
    min_avg_incorrect_guesses = float('inf')
    best_prompt_details = {}

    for p_type, data in summary.items():
        if data["games_played"] > 0:
            solve_rate = data["solved"] / data["games_played"]
            current_avg_incorrect_guesses = (data["total_incorrect_guesses_on_solve"] / data["solved"]) if data["solved"] > 0 else float('inf')

            if solve_rate > max_solve_rate:
                max_solve_rate = solve_rate
                min_avg_incorrect_guesses = current_avg_incorrect_guesses
                best_prompt = p_type
                best_prompt_details = {
                    "prompt_type": p_type,
                    "solve_rate": f"{solve_rate*100:.2f}%",
                    "average_incorrect_guesses_on_solve": current_avg_incorrect_guesses
                }
            elif solve_rate == max_solve_rate:
                if current_avg_incorrect_guesses < min_avg_incorrect_guesses:
                    min_avg_incorrect_guesses = current_avg_incorrect_guesses
                    best_prompt = p_type
                    best_prompt_details = {
                        "prompt_type": p_type,
                        "solve_rate": f"{solve_rate*100:.2f}%",
                        "average_incorrect_guesses_on_solve": current_avg_incorrect_guesses
                    }

    if best_prompt:
        analysis_output["best_performing_prompt"] = best_prompt_details
        analysis_output["best_performing_prompt"]["analysis_based_on_runs"] = NUM_EXPERIMENT_RUNS
    else:
        analysis_output["best_performing_prompt"] = "Could not determine a best performing prompt type."

    with open("analyzed_results_hangman.json", "w") as f:
        json.dump(analysis_output, f, indent=4)

    print("Hangman analysis complete. Results saved to 'analyzed_results_hangman.json'.")

# --- Main Execution ---
if __name__ == "__main__":
    all_experiment_results_across_runs = []
    
    print(f"--- Running {NUM_EXPERIMENT_RUNS} Hangman experiments ---")

    for i in range(NUM_EXPERIMENT_RUNS):
        # Choose a random secret word for each run
        secret_word_for_run = random.choice(wordlist.VALID_WORDS).lower()
        print(f"\n--- Experiment Run {i+1}/{NUM_EXPERIMENT_RUNS} with Secret Word: '{secret_word_for_run}' ---")
        
        # Run the experiment for the current secret word across all prompt types
        run_results = run_experiment(secret_word_for_run, PROMPT_TEMPLATES, wordlist.VALID_WORDS)
        all_experiment_results_across_runs.append(run_results)
        print(f"--- Finished Run {i+1} ---")

    # Analyze and display aggregated results
    analyze_results(all_experiment_results_across_runs, NUM_EXPERIMENT_RUNS)

    # Dump all results to results_hangman.json
    with open("results_hangman.json", "w") as f:
        json.dump(all_experiment_results_across_runs, f, indent=2)
    print(f"\nDetailed results from {NUM_EXPERIMENT_RUNS} runs have been dumped to results_hangman.json")