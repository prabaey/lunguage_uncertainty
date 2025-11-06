import trueskill

from collections import defaultdict
import pandas as pd
import random
import time
import json
import os

from modules.compare_hedging_phrases import LLMSentenceComparer

class TrueSkillRanker:
    def __init__(self, mu=25.0, sigma=8.333, beta=4.167, tau=0.083, draw_probability=0.0):
        self.env = trueskill.TrueSkill(mu=mu, sigma=sigma, beta=beta, tau=tau, draw_probability=draw_probability) # intialize TrueSkill environment
        self.ratings = defaultdict(self.env.Rating) # keep track of ratings
        self.init_mu = mu
        self.init_sigma = sigma
    
    def _get_rating(self, phrase):
        return self.ratings[phrase]
    
    def set_rating(self, phrase, mu=25.0, sigma=8.333): 
        self.ratings[phrase] = self.env.Rating(mu=mu, sigma=sigma)
    
    def update_ratings_from_dataframe(self, df, update_phrase=None, player1_col='phrase_1', player2_col='phrase_2', winner_col='chosen_phrase'):

        # sort by time
        df_sorted = df.sort_values(by="time")

        # play the matches one by one, and update the ratings
        for _, row in df_sorted.iterrows():
            p1 = row[player1_col]
            p2 = row[player2_col]
            winner = row[winner_col]

            rating1 = self._get_rating(p1)
            rating2 = self._get_rating(p2)

            if winner == p1:
                new_rating1, new_rating2 = self.env.rate_1vs1(rating1, rating2) # first argument is winner, second argument is loser
            else:
                new_rating2, new_rating1 = self.env.rate_1vs1(rating2, rating1)

            if update_phrase: # if track_phrase != None
                if update_phrase == p1: # only update rating for the phrase you're tracking
                    self.ratings[p1] = new_rating1
                elif update_phrase == p2: 
                    self.ratings[p2] = new_rating2
            else:
                self.ratings[p1] = new_rating1
                self.ratings[p2] = new_rating2

    def update_rating(self, p1, p2, winner, update_phrase=None): 

        rating1 = self._get_rating(p1)
        rating2 = self._get_rating(p2)
        if winner == p1: 
            new_rating1, new_rating2 = self.env.rate_1vs1(rating1, rating2) # first argument is winner, second argument is loser
        else: 
            new_rating2, new_rating1 = self.env.rate_1vs1(rating2, rating1) # first argument is winner, second argument is loser

        if update_phrase: # if track_phrase != None
            if update_phrase == p1: # only update rating for this phrase
                self.ratings[p1] = new_rating1
            elif update_phrase == p2: 
                self.ratings[p2] = new_rating2
        else:
            self.ratings[p1] = new_rating1
            self.ratings[p2] = new_rating2

    def get_ranking(self, conservative=False):
        def conservative_score(rating): # conservative estimation of skill: extremely likely that player's actual skill is higher
            return rating.mu - 3 * rating.sigma

        if conservative:
            ranked = sorted(self.ratings.items(), key=lambda x: conservative_score(x[1]), reverse=True)
        else: 
            ranked = sorted(self.ratings.items(), key=lambda x: x[1].mu, reverse=True)
        return pd.DataFrame([
            {
                'phrase': phrase,
                'mu': rating.mu,
                'sigma': rating.sigma,
                'score': conservative_score(rating) if conservative else rating.mu
            }
            for phrase, rating in ranked
        ])

    def get_rating(self, phrase):
        """Get (mu, sigma) tuple for a phrase."""
        rating = self.ratings.get(phrase)
        return (rating.mu, rating.sigma) if rating else (None, None)
    
    def copy(self): 
        """Create deep copy of this ranker"""
        new_ranker = TrueSkillRanker()
        for phrase in self.ratings:
            mu = self.ratings[phrase].mu
            sigma = self.ratings[phrase].sigma
            new_ranker.set_rating(phrase, mu, sigma)
        return new_ranker

class SentenceRanker: 

    def __init__(self, reference, vocab, llm_sent_comparer, llm_log_path, max_steps, max_plays, patience, all_LLMs_cutoff, seed): 
        # initialize Ranker object with reference ranking
        self.ranker = self.init_ranker(reference)

        # initialize LLM sentence comparer
        self.llm_sent_comparer = llm_sent_comparer
        self.llm_log_path = llm_log_path

        # vocab contains vocabulary phrases included in reference ranking and their example sentences
        self.vocab = vocab 

        # algorithm settings
        self.max_steps = max_steps
        self.max_plays = max_plays
        self.patience = patience
        self.seed = seed
        self.all_LLMs_cutoff = all_LLMs_cutoff

    def init_ranker(self, reference):
        # Initialize fresh ranker object with reference scores
        ranker = TrueSkillRanker()
        for phrase in reference.index:
            mu = reference.loc[phrase]["mu"]
            sigma = reference.loc[phrase]["sigma"]
            ranker.set_rating(phrase, mu, sigma)
        return ranker

    def play_match(self, sentence, opponent, step, llm_name, max_comp=5):

        # Choose random opponent
        opp_choices = self.vocab[opponent]
        opp_choice = random.choice(opp_choices)
        opp_sent = opp_choice["sentence"].replace(opp_choice["entity"], "<finding>")
        
        # Call LLM to choose the winner
        valid = False
        n_comp = 0
        total_cost = 0
        while not valid and n_comp < max_comp:
            time.sleep(0.05)
            _, winner, cost = self.llm_sent_comparer.compare_sentences(sentence, opp_sent, llm_name)
            winner = winner.strip('"')
            if winner == "sentence_1" or winner == "sentence_2": 
                valid = True
            n_comp += 1
            total_cost += cost
        if n_comp == max_comp: 
            print("Error in LLM call! Inspect logs")

        # Log the last LLM call
        log_entry = {
            'step': step,
            'llm_name': llm_name,
            'sentence': sentence,
            'opp_sentence': opp_sent,
            'opp': opponent,
            'winner': winner, 
            'cost': total_cost
        }
        with open(self.llm_log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')

        resp = "SENT" if winner == "sentence_1" else opponent
        
        return resp, total_cost

    def rank_unseen_sentence(self, sentence): 

        # Set seed
        random.seed(self.seed)
        
        # Keep track of how many times you played each opponent
        n_play_opp = {opp: 0 for opp in self.vocab}

        # Initialize rating for new sentence
        self.ranker.set_rating("SENT")

        # Play games with new sentence to determine ranking among established bleuprint
        step = 0
        stop_reached = False
        logs = []
        last_rank = None
        constant_rank_count = 0
        while not stop_reached and step < self.max_steps:

            rating_sent = self.ranker._get_rating("SENT")
            chosen_opp = None
            best_score = 0.0

            # Find the opponent with maximal draw probability
            for opp in self.vocab: 

                # Only allow opponents that have not exhausted the maximal nr. of plays
                if n_play_opp[opp] < self.max_plays:

                    # Calculate the draw probability
                    rating_opp = self.ranker._get_rating(opp)
                    draw_prob = trueskill.quality_1vs1(rating_sent, rating_opp)

                    # Choose opponent with highest draw probability
                    if draw_prob > best_score:
                        best_score = draw_prob
                        chosen_opp = opp

            # Play match with chosen opponent
            if step < self.all_LLMs_cutoff: 

                # At the start, all comparisons are played by each of the 3 LLMs
                time.sleep(1)
                win_GPT, cost_GPT = self.play_match(sentence, chosen_opp, step, "GPT")
                win_gemini, cost_gemini = self.play_match(sentence, chosen_opp, step, "Gemini")
                win_claude, cost_claude = self.play_match(sentence, chosen_opp, step, "Claude")
                win_medgemma, cost_medgemma = self.play_match(sentence, chosen_opp, step, "MedGemma")
                llm_name = "all"
                cost = cost_GPT + cost_gemini + cost_claude + cost_medgemma
                winner = [win_GPT, win_gemini, win_claude, win_medgemma]
                n_play_opp[chosen_opp] += 1
            else: 

                # After the number of steps exceeds the cutoff, a random LLM is chosen
                llm_name = random.choice(["GPT", "Gemini", "Claude", "MedGemma"])
                win, cost = self.play_match(sentence, chosen_opp, step, llm_name)
                winner = [win]
                n_play_opp[chosen_opp] += 1

            # Update ranking with outcome of match
            for win in winner:
                self.ranker.update_rating("SENT", chosen_opp, win, update_phrase="SENT")

            # Calculate updated ranking
            df_rank = self.ranker.get_ranking(conservative=False)
            rank_sent = df_rank[df_rank["phrase"] == "SENT"].index[0]
            rank_opp = df_rank[df_rank["phrase"] == chosen_opp].index[0]
            score_sent = df_rank[df_rank["phrase"] == "SENT"]["score"].iloc[0]
            sigma_sent = df_rank[df_rank["phrase"] == "SENT"]["sigma"].iloc[0]

            # Check if rank stayed the same
            if rank_sent == last_rank:
                constant_rank_count += 1
            else: 
                constant_rank_count = 1
            last_rank = rank_sent

            # If rank stayed the same for patience steps, then finish ranking
            if constant_rank_count >= self.patience:
                stop_reached = True

            # Log information
            log_dict = {
                "step": step,
                "opponent": chosen_opp,
                "draw_prob": best_score,
                "winner": winner,
                "rank_sent": rank_sent, 
                "rank_opp": rank_opp, 
                "score_sent": score_sent, 
                "sigma_sent": sigma_sent,
                "cost": cost, 
                "llm_name": llm_name
            }
            logs.append(log_dict)

            step += 1

        return pd.DataFrame(logs), rank_sent
    
def rank_all_sentences():

    # cases where <finding> cannot be automatically matched are handled manually
    subs_dict = {168: "it", 182: "inflammatory", 760: "cardiac", 813: "cardiac", 2994: "this", 
             3347: "cardiac", 5109: "interstitial", 5239: "hilar", 5854: "cardiomediastinal sillouhette",
             7143: "effusions", 7555: "cardiac", 9633: "interstitial", 10588: "cardiac", 10946: "pathologic pleural involvement", 
             11789: "ventricle", 12098: "these", 12774: "cardiac"}

    # loop over all finding-sentence pairs in the tentative Lunguage dataset
    n_sent = 2100
    all_LLMs_cutoff = 10
    result_log = f"../../data_resources/ranking_log/rank_sentence_log.jsonl"
    master_seed = 42
    random.seed(master_seed)

    # read reference ranking and vocab
    reference = pd.read_csv("../../data_resources/reference_ranking.csv")
    reference.set_index("phrase", inplace=True)
    with open("../../data_resources/hedging_phrase_vocab.json", 'r', encoding='utf-8') as f:
        vocab = json.load(f)

    # initialize LLM object 
    gemini_project="your-gemini-project-here"
    openai_endpoint="your-azure-openai-endpoint-here"
    openai_key="your-azure-openai-key-here"
    anthropic_key="your-anthropic-key-here"
    huggingface_token="your-huggingface-token-here"
    llm_sent_comparer = LLMSentenceComparer(gemini_project, openai_endpoint, openai_key, anthropic_key, huggingface_token)

    # load in tentative Lunguage finding-sentence pair
    df_gold = pd.read_csv("../../data_resources/lunguage/Lunguage.csv", index_col=0)
    df_gold = df_gold[df_gold["section"] != "hist"]
    df_tent = df_gold[df_gold["dx_certainty"] == "tentative"]

    # loop over all tentative entity-sentence pairs
    loop_obj = []
    for i, row in df_tent.iterrows(): 
        loop_obj.append({"entity": row["ent"], "sentence": row["sent"]})
    
    # load in logging file as a dictionary, add results to the dictionary as you go along
    dict_completed = {}
    if os.path.exists(result_log):
        with open(result_log, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                key = obj.get("idx")
                dict_completed[key] = obj

    i = 0
    for item in loop_obj:

        # add to counter
        i += 1

        # find all indices in dataset with the same entity-sentence pair
        entity = item["entity"]
        sentence = item["sentence"]
        sel_idxs = list(df_tent.loc[(df_tent["ent"] == entity) & (df_tent["sent"] == sentence)].index)

        # check if entity-sentence pair has already been ranked
        rank_sent = False
        for idx in sel_idxs: 
            if idx not in dict_completed: 
                rank_sent = True
                break

        if rank_sent: 

            # mask <finding> in candidate sentence. use substitution dict for the few cases where entity not found in sentence
            if entity not in sentence: 
                subs_name = subs_dict[idx]
                masked_sentence = sentence.replace(subs_name, "<finding>")
                print(idx, masked_sentence)
            else: 
                masked_sentence = sentence.replace(entity, "<finding>")

            # initialize sentence ranker
            max_steps = 100
            max_plays = 5
            patience = 10
            run_seed = random.randint(0, 10000)
            llm_log_path = f"../../data_resources/ranking_log/llm_logs/{sel_idxs[0]}.jsonl"
            sent_ranker = SentenceRanker(reference, vocab, llm_sent_comparer, llm_log_path, max_steps, max_plays, patience, all_LLMs_cutoff, seed=run_seed)

            # rank candidate sentence
            logs, rank = sent_ranker.rank_unseen_sentence(masked_sentence)

            # store logs as csv
            for idx in sel_idxs:
                logs.to_csv(f"../../data_resources/ranking_log/ranker_logs/{idx}.csv", index=False)

            # store LLM logs for other indices
            for idx in sel_idxs[1:]: 
                with open(llm_log_path, "r", encoding="utf-8") as src, open(f"../../data_resources/ranking_log/llm_logs/{idx}.jsonl", "w", encoding="utf-8") as dst:
                    for line in src:
                        dst.write(line)

            # store result in dict and in logging file as you go along
            for idx in sel_idxs: 
                log_entry = {
                    'idx': idx,
                    'entity': entity, 
                    'sentence': sentence, 
                    'rank': rank.item(), 
                    'score': logs.iloc[-1]["score_sent"].item(), 
                    'sigma': logs.iloc[-1]["sigma_sent"].item(), 
                    "cost": logs["cost"].sum(), 
                    "steps": len(logs) 
                }
                dict_completed[idx] = log_entry
                with open(result_log, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry) + '\n')

        else: 
            print(f"already ranked! skipping {sel_idxs}")

        if i >= n_sent: 
            break

    llm_sent_comparer.httpx_client.close()
    
if __name__ == "__main__":

    rank_all_sentences()