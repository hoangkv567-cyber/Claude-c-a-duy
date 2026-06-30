import unittest
import json
import tempfile
import os
from src.mapping import SymptomToSyndromeMapper

class TestSymptomToSyndromeMapper(unittest.TestCase):
    def setUp(self):
        # Create a temporary mapping file for testing
        self.temp_dir = tempfile.TemporaryDirectory()
        self.mapping_data = {
            "Lưỡi bệu có dấu răng": ["Tỳ vị hư nhược", "Đàm thấp uẩn phế"],
            "Lưỡi đỏ": ["Can uất hóa hỏa", "Âm hư hỏa vượng"],
            "Mặt vàng": ["Tỳ vị hư nhược"]
        }
        self.mapping_file_path = os.path.join(self.temp_dir.name, "test_mapping.json")
        with open(self.mapping_file_path, "w", encoding="utf-8") as f:
            json.dump(self.mapping_data, f, ensure_ascii=False, indent=2)

    def tearDown(self):
        # Clean up temporary directory and files
        self.temp_dir.cleanup()

    def test_load_mapping(self):
        # Test loading mapping file successfully
        mapper = SymptomToSyndromeMapper(self.mapping_file_path)
        self.assertEqual(mapper.mapping, self.mapping_data)

    def test_load_nonexistent_file(self):
        # Test loading a nonexistent file uses empty mapping
        mapper = SymptomToSyndromeMapper("nonexistent_file.json")
        self.assertEqual(mapper.mapping, {})

    def test_map_single_symptom(self):
        # Test mapping a single valid symptom
        mapper = SymptomToSyndromeMapper(self.mapping_file_path)
        result = mapper.map_symptoms_to_syndromes(["Lưỡi đỏ"])
        self.assertCountEqual(result, ["Can uất hóa hỏa", "Âm hư hỏa vượng"])

    def test_map_multiple_symptoms(self):
        # Test mapping multiple symptoms returns unique set of syndromes
        mapper = SymptomToSyndromeMapper(self.mapping_file_path)
        result = mapper.map_symptoms_to_syndromes(["Lưỡi bệu có dấu răng", "Mặt vàng"])
        self.assertCountEqual(result, ["Tỳ vị hư nhược", "Đàm thấp uẩn phế"])

    def test_map_unknown_symptom(self):
        # Test mapping an unknown symptom is ignored
        mapper = SymptomToSyndromeMapper(self.mapping_file_path)
        result = mapper.map_symptoms_to_syndromes(["Triệu chứng lạ", "Lưỡi đỏ"])
        self.assertCountEqual(result, ["Can uất hóa hỏa", "Âm hư hỏa vượng"])

if __name__ == "__main__":
    unittest.main()
