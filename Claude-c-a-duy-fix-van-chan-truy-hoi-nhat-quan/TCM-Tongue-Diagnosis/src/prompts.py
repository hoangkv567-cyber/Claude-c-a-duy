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
Analyze this face image carefully and output ONLY a JSON array of symptoms.

IMPORTANT: Check the following features step by step:
1. Face color: red, pale, yellow, greenish, dark
2. Swelling: swollen, normal
3. Rashes or spots: yes, no

Choose ALL symptoms that match from this list:
{symptom_list}

Output format: ["symptom1", "symptom2", ...]
Do not write any explanations or other text.
"""
