TONGUE_PROMPT_TEMPLATE = """
Act as a Traditional Chinese Medicine expert with 20 years of experience in tongue diagnosis (Vọng chẩn).
Analyze this tongue image carefully and output ONLY a JSON array of symptoms.

IMPORTANT: Check the following features step by step:
1. Tongue color: red, pale, purple, normal pink
2. Tongue coating: white, yellow, thin, thick, peeled, no coating
3. Tongue shape: swollen, tooth marks, cracked, thin, normal

Choose ALL symptoms that match from this list:
{symptom_list}

Output format: ["symptom1", "symptom2", ...]
Do not write any explanations or other text.
"""

SYNDROME_PROMPT_TEMPLATE = """
Based on the following observed symptoms from a tongue image: {symptoms}
Suggest the most likely TCM syndrome (hội chứng) from this list: {syndrome_list}
Output ONLY the syndrome name as a JSON string. Example: "Tỳ vị hư nhược"
"""

FACE_PROMPT_TEMPLATE = """
Act as a Traditional Chinese Medicine expert with 20 years of experience in face diagnosis (Vọng chẩn - nhìn sắc mặt).
Analyze this face image carefully and write a concise, professional description of the patient's facial features in English.

Please describe:
1. Complexion / Facial color (e.g., pale, sallow, flushed red, dull/dark, yellow, green...).
2. Spirit / Expression / Shen (e.g., fatigue, lack of spirit, dull eyes, alert/normal...).
3. Other facial features or abnormalities (e.g., puffiness/swelling, dark circles under eyes, spots, rashes, moles...).

Write a concise description in English (1-2 sentences). Do not use JSON or lists. Just write the description directly.
"""
