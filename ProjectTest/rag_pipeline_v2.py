#!/usr/bin/env python3

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from embeddings import VectorStore

log = logging.getLogger(__name__)


class RAGPipeline:

    def __init__(self, vectorstore_dir, model_dir, device="cpu", **kwargs):

        print("Running CLEAN CPU pipeline")

        # Load vector database
        self.vectorstore = VectorStore(
            vectorstore_dir,
            device="cpu"
        )

        # Load TinyLlama
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="cpu",
            torch_dtype=torch.float32
        )

    def recommend(self, query, vocab_list=None, **kwargs):

        # ----------------------------------------
        # Retrieve candidate articles
        # ----------------------------------------

        results = self.vectorstore.search(
            query,
            top_k=3
        )

        if not results:

            return {
                "recommended_title": "No article found",

                "summary":
                    "No matching article was retrieved.",

                "why_good_next_read": "",

                "difficulty_rating": 0,

                "confidence_score": 0.0,

                "new_vocabulary": [],

                "_retrieved_articles": [],

                "_query": query
            }

        # ----------------------------------------
        # Select top article
        # ----------------------------------------

        top_article = results[0]

        article_title = top_article.get(
            "title",
            "Recommended Article"
        )

        article_text = top_article.get(
            "text",
            ""
        )

        # ----------------------------------------
        # Generate summary with TinyLlama
        # ----------------------------------------

        prompt = f"""
You are a vocabulary recommendation assistant.

Summarize this article in 2-3 sentences.

TOPIC:
{query}

ARTICLE:
{article_text[:2500]}

SUMMARY:
"""

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        with torch.no_grad():

            output = self.model.generate(
                **inputs,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.7
            )

        response = self.tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        # ----------------------------------------
        # Clean generated response
        # ----------------------------------------

        if "SUMMARY:" in response:

            cleaned = response.split(
                "SUMMARY:"
            )[-1].strip()

        else:

            cleaned = response[-600:].strip()

        # ----------------------------------------
        # Vocabulary analysis
        # ----------------------------------------

        analysis = top_article.get(
            "analysis",
            {}
        )

        vocab_words = analysis.get(
            "new_words",
            []
        )

        coverage_ratio = analysis.get(
            "coverage_ratio",
            0.95
        )

        # ----------------------------------------
        # Filter vocabulary words
        # ----------------------------------------

        filtered_words = []

        seen = set()

        for word in vocab_words:

            if not isinstance(word, str):
                continue

            word = word.lower().strip()

            # Remove short tokens
            if len(word) < 4:
                continue

            # Remove non alphabetic tokens
            if not word.isalpha():
                continue

            # Remove duplicates
            if word in seen:
                continue

            seen.add(word)

            filtered_words.append(word)

        # ----------------------------------------
        # Remove already known vocabulary
        # ----------------------------------------

        if vocab_list is not None:

            known_words = {
                w.lower()
                for w in vocab_list.known_words
            }

            filtered_words = [
                w for w in filtered_words
                if w not in known_words
            ]

        # Prevent huge vocabulary explosions
        filtered_words = filtered_words[:50]

        # ----------------------------------------
        # Build vocabulary list
        # ----------------------------------------

        new_vocabulary = []

        for word in filtered_words:

            new_vocabulary.append({

                "word": word,

                "definition":
                    f"{word} is an important vocabulary term related to this article.",

                "example":
                    f"The article discusses the concept of {word}."

            })

        # ----------------------------------------
        # Dynamic difficulty score
        # ----------------------------------------

        difficulty = int(
            ((1.0 - coverage_ratio) * 100) / 10
        )

        difficulty = max(
            1,
            min(10, difficulty)
        )

        # ----------------------------------------
        # Dynamic confidence score
        # ----------------------------------------

        confidence = min(
            0.99,
            0.70 + (coverage_ratio * 0.3)
        )

        # ----------------------------------------
        # Final response
        # ----------------------------------------

        return {

            "recommended_title": article_title,

            "summary": cleaned,

            "why_good_next_read":
                f"This article matches the topic '{query}' "
                f"and fits the vocabulary recommendation system.",

            "difficulty_rating": difficulty,

            "confidence_score": round(
                confidence,
                2
            ),

            "new_vocabulary": new_vocabulary,

            "_retrieved_articles": results,

            "_query": query
        }
