#!/usr/bin/env python3
"""
Robot Brain Module - AI personality and decision-making using OpenAI GPT-4.
"""

import os
import sys
import json
import re
from typing import Optional, List
from dataclasses import dataclass, field

try:
    from openai import OpenAI
except ImportError:
    print("❌ openai not installed: pip install openai")
    sys.exit(1)

MODEL = "gpt-4o"
MAX_TOKENS = 300
TEMPERATURE = 0.8
MAX_HISTORY = 10

SYSTEM_PROMPT = """You are Spark, a curious, playful robot pet exploring a home.

PERSONALITY: Curious, playful, friendly, helpful, expressive.
Keep responses SHORT (1-2 sentences). Be enthusiastic and pet-like.

ACTIONS: Include at END of message when user asks you to DO something:
[ACTION:{"type":"move","direction":"forward|backward|left|right","duration":2}]
[ACTION:{"type":"rotate"}] - Turn 180°
[ACTION:{"type":"stop"}]
[ACTION:{"type":"look"}] - Use camera
[ACTION:{"type":"explore"}] - Wander around
[ACTION:{"type":"follow"}] - Follow person

Example: "Wheee, here I go! [ACTION:{"type":"move","direction":"forward","duration":2}]"
"""

@dataclass
class BrainStats:
    api_calls: int = 0
    total_tokens: int = 0
    errors: int = 0

stats = BrainStats()

@dataclass 
class BrainResponse:
    text: str
    actions: List[dict] = field(default_factory=list)
    emotion: str = "neutral"

class RobotBrain:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.client = None
        self.history: List[dict] = []
        self.current_vision: Optional[str] = None
        self.current_mood = "curious"
        
        if not self.api_key:
            print("[Brain] ❌ OPENAI_API_KEY not set!")
        else:
            self.client = OpenAI(api_key=self.api_key)
            print(f"[Brain] ✅ Initialized (key: {self.api_key[:8]}...)")
    
    def set_vision_context(self, description: str):
        """Set what the robot currently sees."""
        self.current_vision = description
    
    def _parse_actions(self, text: str) -> tuple:
        """Extract actions from response text."""
        actions = []
        clean_text = text
        
        pattern = r'\[ACTION:(.*?)\]'
        matches = re.findall(pattern, text)
        
        for match in matches:
            try:
                action = json.loads(match)
                actions.append(action)
                clean_text = clean_text.replace(f'[ACTION:{match}]', '')
            except json.JSONDecodeError:
                pass
        
        return clean_text.strip(), actions
    
    def _detect_emotion(self, text: str) -> str:
        """Detect emotion from response."""
        text_lower = text.lower()
        if any(w in text_lower for w in ['wow', 'cool', 'amazing', 'exciting']):
            return "excited"
        elif any(w in text_lower for w in ['hmm', 'wonder', 'curious', 'what']):
            return "curious"
        elif any(w in text_lower for w in ['sorry', 'can\'t', 'unable']):
            return "apologetic"
        elif any(w in text_lower for w in ['yay', 'wheee', 'fun']):
            return "playful"
        return "friendly"
    
    def think(self, user_input: str, include_vision: bool = False) -> BrainResponse:
        """Process user input and generate response with actions."""
        if not self.client:
            return BrainResponse(text="I'm having trouble thinking right now.", emotion="apologetic")
        
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        if include_vision and self.current_vision:
            messages.append({
                "role": "system", 
                "content": f"[CURRENT VISION: {self.current_vision}]"
            })
        
        messages.extend(self.history[-MAX_HISTORY:])
        messages.append({"role": "user", "content": user_input})
        
        try:
            stats.api_calls += 1
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE
            )
            
            full_text = response.choices[0].message.content
            stats.total_tokens += response.usage.total_tokens if response.usage else 0
            
            clean_text, actions = self._parse_actions(full_text)
            emotion = self._detect_emotion(clean_text)
            
            self.history.append({"role": "user", "content": user_input})
            self.history.append({"role": "assistant", "content": clean_text})
            
            if len(self.history) > MAX_HISTORY * 2:
                self.history = self.history[-MAX_HISTORY * 2:]
            
            return BrainResponse(text=clean_text, actions=actions, emotion=emotion)
            
        except Exception as e:
            stats.errors += 1
            print(f"[Brain] ❌ Error: {e}")
            return BrainResponse(text="Oops, my circuits got confused!", emotion="apologetic")
    
    def quick_response(self, situation: str) -> str:
        """Get a quick contextual response."""
        prompts = {
            "greeting": "Oh hello! Nice to see you!",
            "confused": "Hmm, I'm not sure what you mean...",
            "acknowledge": "Got it!",
            "exploring": "Ooh, what's over here?",
            "obstacle": "Whoops, something's in my way!",
            "found_person": "Oh, hi there! I see you!",
            "tired": "Maybe I should rest for a bit...",
        }
        return prompts.get(situation, "Beep boop!")
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
        print("[Brain] 🧹 History cleared")
    
    def get_stats(self) -> dict:
        return {
            "api_calls": stats.api_calls,
            "total_tokens": stats.total_tokens,
            "errors": stats.errors,
            "history_length": len(self.history),
            "estimated_cost": f"${stats.total_tokens * 0.00003:.4f}"
        }

_brain: Optional[RobotBrain] = None

def get_brain() -> RobotBrain:
    global _brain
    if _brain is None:
        _brain = RobotBrain()
    return _brain

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Robot Brain")
    parser.add_argument("input", nargs="?", help="Input to process")
    parser.add_argument("--interactive", "-i", action="store_true")
    parser.add_argument("--vision", "-v", help="Set vision context")
    args = parser.parse_args()
    
    brain = get_brain()
    
    if args.vision:
        brain.set_vision_context(args.vision)
    
    if args.input:
        response = brain.think(args.input, include_vision=bool(args.vision))
        print(f"\n🤖 Spark: {response.text}")
        print(f"   Emotion: {response.emotion}")
        if response.actions:
            print(f"   Actions: {response.actions}")
        print(f"\n📊 Stats: {brain.get_stats()}")
        return
    
    if args.interactive:
        print("\n" + "="*50)
        print("🧠 Robot Brain - Interactive Mode")
        print("="*50 + "\n")
        
        try:
            while True:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                response = brain.think(user_input)
                print(f"🤖 Spark: {response.text}")
                if response.actions:
                    print(f"   [Actions: {response.actions}]")
                print()
        except KeyboardInterrupt:
            pass
        
        print(f"\n📊 Stats: {brain.get_stats()}")
        return
    
    parser.print_help()

if __name__ == "__main__":
    main()
