TONGUE_PROMPT_TEMPLATE = """
Act as a Traditional Chinese Medicine expert with 20 years of experience in tongue diagnosis (Vọng chẩn - xem lưỡi).
Analyze this tongue image carefully and write a concise, professional description of the patient's tongue features in English.

Please describe:
1. Tongue body color (e.g., pale, red, deep red, purple, normal pink...).
2. Tongue coating color and texture (e.g., white or yellow coating, thin or thick coating, greasy/sticky, dry, peeled, or no coating...).
3. Tongue shape and features (e.g., swollen, thin, cracked, tooth marks/scalloped edges...).

Write a concise description in English (1-2 sentences). Do not use JSON or lists. Just write the description directly.
"""

SYNDROME_PROMPT_TEMPLATE = """
Based on the following observed symptoms from a tongue image: {symptoms}
Suggest the most likely TCM syndrome (hội chứng) from this list: {syndrome_list}
Output ONLY the syndrome name as a JSON string. Example: "Tỳ vị hư nhược"
"""

FACE_PROMPT_TEMPLATE = """
Act as a TCM face diagnosis expert. Describe the patient's face objectively based on these exact questions:
1. What is the complexion/skin color? (e.g., sallow, pale, yellowish-pale, dull, flushed red, or normal/healthy pink).
2. Are there wrinkles, lines, or creases on the forehead or between the eyebrows (Ấn Đường)? (Describe them if present).
3. Are there any small moles, spots, or marks on the chin, cheeks, forehead, or neck? (Describe where they are).
4. What are the eyebrows like? (e.g., sparse, thick, symmetrical, or asymmetrical).
5. Are there dark circles, puffiness under the eyes, or visible laugh lines?

Write a concise, factual description in English (2-3 sentences) combining these details. Do not use healthy/smooth/rosy templates unless they actually apply. Be highly realistic and objective.
"""
