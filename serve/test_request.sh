#!/usr/bin/env bash
# Quick check once the server runs. Model name = the adapter name given in --lora-modules.
curl -s http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $VLLM_API_KEY" -H "Content-Type: application/json" \
  -d '{"model": "revalo-waste", "temperature": 0, "max_tokens": 200,
       "messages": [{"role": "user", "content": "Task: waste_sort. Classify the waste item. Return ONLY a JSON object with keys: category (organic | plastic_rigid | plastic_film | paper_cardboard | glass | metal | ewaste | hazardous | textile | residual), valorisable (true/false), hazardous (true/false), route (what to do with it, in the user'"'"'s language: French for French, English for English or Pidgin).\nQuestion: Je fais quoi avec des piles usagées ?"}]}'
