"""
Processes a directory of plain text files into a single JSONL file with
detailed linguistic and temporal annotations. Handles dates in the document header.
"""

import os
import spacy
import json
import re
import argparse
import glob
from datetime import datetime
from dateutil.parser import parse, ParserError
from tqdm import tqdm

def process_document(doc_id, text_content, nlp_model):
    """
    Processes a raw text string to extract metadata and parse sentences.
    """
    try:
        # --- 1. Smarter Publication Time Extraction ---
        pub_time_str = None
        header_text = "\n".join(text_content.strip().split('\n')[:5])

        date_regex = r"\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
        date_match = re.search(date_regex, header_text)

        if not date_match:
            date_match = re.search(r"\[(.*?)\]", header_text)

        if date_match:
            try:
                date_text = date_match.group(0) if date_match.re.pattern == date_regex else date_match.group(1)
                default_date = datetime(1, 1, 1).replace(day=1)
                dt_obj = parse(date_text, default=default_date)
                pub_time_str = dt_obj.isoformat() + "+00:00"
            except (ValueError, TypeError, AttributeError):
                pub_time_str = None

        # --- Basic Metadata Extraction ---
        title_match = re.search(r"^\d+\.\s*(.*)", text_content, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else f"Document_{doc_id}"

        body_match = re.search(r"(DEAR SIR,|MY DEAR.*?)(.*?)(Yours sincerely,|Yours faithfully,)", text_content, re.DOTALL | re.IGNORECASE)
        if body_match:
            body_text = body_match.group(2).strip()
        else:
            lines = text_content.strip().split('\n')
            body_text = "\n".join(lines[5:])
        body_text = re.sub(r'\s+', ' ', body_text).strip()

        if not body_text:
            return None

        # --- Process the full body with SpaCy ---
        doc = nlp_model(body_text)
        processed_sentences = []

        for sent in doc.sents:
            sentence_level_time = None
            token_time_map = {}

            for ent in sent.ents:
                if ent.label_ == "DATE":
                    try:
                        parsed_time = parse(ent.text)
                        time_val = parsed_time.isoformat()
                        time_format = "YYYY-MM-DD"
                        if not sentence_level_time:
                            sentence_level_time = time_val
                        for token in ent:
                            token_time_map[token.i] = (time_val, time_format)
                    except (ParserError, ValueError):
                        continue

            sentence_data = {
                "raw": sent.text,
                "tokens": {
                    "keys": ["dep", "head", "lemma", "ner_iob", "ner_type", "pos", "raw", "time", "time_format"],
                    "data": []
                },
                "time": sentence_level_time,
                "pub_time": pub_time_str
            }

            token_list = []
            for token in sent:
                time_val, time_format = token_time_map.get(token.i, (None, None))
                token_list.append([
                    token.dep_, token.head.i - sent.start, token.lemma_,
                    token.ent_iob_, token.ent_type_, token.pos_, token.text,
                    time_val, time_format
                ])

            sentence_data["tokens"]["data"] = token_list
            processed_sentences.append(sentence_data)

        final_json = {
            "title": title,
            "text": text_content.strip(),
            "time": pub_time_str,
            "id": str(doc_id),
            "sentences": processed_sentences
        }
        return final_json

    except Exception as e:
        print(f"Error processing document {doc_id}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Process a directory of text files into a single JSONL file.")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the directory containing .txt files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output .jsonl file.")
    parser.add_argument("--model", type=str, default="en_core_web_sm", help="Name of the spaCy model to use.")
    args = parser.parse_args()

    print(f"Loading spaCy model '{args.model}'...")
    try:
        nlp = spacy.load(args.model)
    except OSError:
        print(f"Error: spaCy model '{args.model}' not found. Please run: python -m spacy download {args.model}")
        return
    print("Model loaded successfully.")

    # --- List of allowed filenames ---
    allowed_files = set([
        "volume45_book_134.txt",
        "volume47_book_365.txt",
        "volume92_book_384.txt",
        "volume47_book_350.txt",
        "volume46_book_130.txt",
        "volume47_book_41.txt",
        "volume44_book_531.txt",
        "volume45_book_23.txt",
        "volume44_book_473.txt",
        "volume45_book_331.txt",
        "volume44_book_483.txt",
        "volume92_book_453.txt",
        "volume47_book_247.txt",
        "volume46_book_372.txt",
        "volume47_book_388.txt",
        "volume45_book_156.txt",
        "volume46_book_13.txt",
        "volume47_book_293.txt",
        "volume46_book_40.txt",
        "volume47_book_321.txt",
        "volume47_book_185.txt",
        "volume46_book_186.txt",
        "volume47_book_111.txt",
        "volume47_book_326.txt",
        "volume47_book_176.txt",
        "volume47_book_133.txt",
        "volume46_book_71.txt",
        "volume45_book_50.txt",
        "volume47_book_74.txt",
        "volume45_book_83.txt",
        "volume46_book_104.txt",
        "volume46_book_89.txt",
        "volume46_book_346.txt",
        "volume45_book_320.txt",
        "volume47_book_339.txt",
        "volume46_book_316.txt",
        "volume45_book_112.txt",
        "volume45_book_101.txt",
        "volume44_book_443.txt",
        "volume46_book_220.txt",
        "volume46_book_82.txt",
        "volume47_book_85.txt",
        "volume44_book_448.txt",
        "volume47_book_405.txt",
        "volume47_book_62.txt",
        "volume47_book_370.txt",
        "volume92_book_434.txt",
        "volume44_book_432.txt",
        "volume46_book_29.txt",
        "volume45_book_82.txt",
        "volume47_book_338.txt",
        "volume45_book_91.txt",
        "volume47_book_49.txt",
        "volume47_book_134.txt",
        "volume46_book_497.txt",
        "volume47_book_273.txt",
        "volume92_book_451.txt",
        "volume45_book_125.txt",
        "volume45_book_157.txt",
        "volume46_book_277.txt",
        "volume45_book_138.txt",
        "volume45_book_257.txt",
        "volume92_book_455.txt",
        "volume47_book_398.txt",
        "volume44_book_526.txt",
        "volume46_book_493.txt",
        "volume46_book_254.txt",
        "volume47_book_89.txt",
        "volume44_book_481.txt",
        "volume46_book_120.txt",
        "volume45_book_117.txt",
        "volume92_book_437.txt",
        "volume46_book_354.txt",
        "volume46_book_435.txt",
        "volume46_book_412.txt",
        "volume45_book_301.txt",
        "volume47_book_17.txt",
        "volume47_book_226.txt",
        "volume47_book_58.txt",
        "volume92_book_466.txt",
        "volume47_book_25.txt",
        "volume45_book_235.txt",
        "volume92_book_430.txt",
        "volume45_book_222.txt",
        "volume44_book_536.txt",
        "volume92_book_459.txt",
        "volume92_book_469.txt",
        "volume45_book_342.txt",
        "volume47_book_26.txt",
        "volume47_book_92.txt",
        "volume45_book_77.txt",
        "volume44_book_518.txt",
        "volume44_book_472.txt",
        "volume45_book_38.txt",
        "volume45_book_115.txt",
        "volume44_book_522.txt",
        "volume46_book_100.txt",
        "volume45_book_191.txt",
        "volume46_book_350.txt",
        "volume45_book_96.txt",
        "volume44_book_514.txt",
        "volume45_book_210.txt",
        "volume47_book_322.txt",
        "volume47_book_216.txt",
        "volume46_book_198.txt",
        "volume45_book_161.txt",
        "volume46_book_136.txt",
        "volume44_book_503.txt",
        "volume46_book_250.txt",
        "volume45_book_303.txt",
        "volume47_book_327.txt",
        "volume47_book_352.txt",
        "volume45_book_86.txt",
        "volume45_book_62.txt",
        "volume47_book_107.txt",
        "volume45_book_135.txt",
        "volume45_book_286.txt",
        "volume46_book_31.txt",
        "volume44_book_429.txt",
        "volume46_book_458.txt",
        "volume45_book_449.txt",
        "volume45_book_308.txt",
        "volume46_book_360.txt",
        "volume47_book_172.txt",
        "volume47_book_424.txt",
        "volume45_book_72.txt",
        "volume47_book_52.txt",
        "volume45_book_234.txt",
        "volume44_book_440.txt",
        "volume47_book_404.txt",
        "volume92_book_471.txt",
        "volume45_book_87.txt",
        "volume47_book_28.txt",
        "volume47_book_399.txt",
        "volume46_book_466.txt",
        "volume45_book_260.txt",
        "volume45_book_8.txt",
        "volume46_book_131.txt",
        "volume46_book_278.txt",
        "volume92_book_398.txt",
        "volume45_book_1.txt",
        "volume45_book_34.txt",
        "volume45_book_46.txt",
        "volume47_book_248.txt",
        "volume92_book_418.txt",
        "volume92_book_408.txt",
        "volume44_book_527.txt",
        "volume44_book_433.txt",
        "volume47_book_64.txt",
        "volume46_book_305.txt",
        "volume46_book_129.txt",
        "volume47_book_97.txt",
        "volume46_book_90.txt",
        "volume44_book_478.txt",
        "volume47_book_402.txt",
        "volume47_book_325.txt",
        "volume46_book_295.txt",
        "volume92_book_427.txt",
        "volume45_book_150.txt",
        "volume45_book_141.txt",
        "volume46_book_348.txt",
        "volume46_book_78.txt",
        "volume47_book_188.txt",
        "volume44_book_457.txt",
        "volume92_book_411.txt",
        "volume45_book_334.txt",
        "volume46_book_161.txt",
        "volume92_book_425.txt",
        "volume92_book_449.txt",
        "volume46_book_122.txt",
        "volume47_book_384.txt",
        "volume47_book_83.txt",
        "volume46_book_185.txt",
        "volume45_book_332.txt",
        "volume45_book_258.txt",
        "volume46_book_414.txt",
        "volume45_book_195.txt",
        "volume47_book_90.txt",
        "volume46_book_477.txt",
        "volume44_book_442.txt",
        "volume45_book_391.txt",
        "volume45_book_88.txt",
        "volume47_book_108.txt",
        "volume47_book_7.txt",
        "volume47_book_122.txt",
        "volume47_book_191.txt",
        "volume46_book_495.txt",
        "volume46_book_368.txt",
        "volume47_book_51.txt",
        "volume45_book_254.txt",
        "volume44_book_499.txt",
        "volume45_book_233.txt",
        "volume92_book_391.txt",
        "volume47_book_43.txt",
        "volume44_book_517.txt",
        "volume45_book_142.txt",
        "volume46_book_273.txt",
        "volume47_book_358.txt",
        "volume44_book_452.txt",
        "volume92_book_394.txt",
        "volume92_book_393.txt",
        "volume47_book_23.txt",
        "volume45_book_123.txt",
        "volume92_book_432.txt",
        "volume92_book_405.txt",
        "volume46_book_9.txt",
        "volume47_book_313.txt",
        "volume46_book_51.txt",
        "volume45_book_128.txt",
        "volume46_book_341.txt",
        "volume45_book_392.txt",
        "volume46_book_470.txt",
        "volume46_book_462.txt",
        "volume46_book_88.txt",
        "volume92_book_392.txt",
        "volume45_book_36.txt",
        "volume46_book_424.txt",
        "volume45_book_43.txt",
        "volume46_book_102.txt",
        "volume44_book_491.txt",
        "volume47_book_428.txt",
        "volume47_book_287.txt",
        "volume45_book_194.txt",
        "volume46_book_279.txt",
        "volume97_book_188.txt",
        "volume46_book_381.txt",
        "volume46_book_312.txt",
        "volume45_book_9.txt",
        "volume46_book_16.txt",
        "volume47_book_160.txt",
        "volume47_book_381.txt",
        "volume92_book_450.txt",
        "volume46_book_371.txt",
        "volume45_book_249.txt",
        "volume47_book_382.txt",
        "volume46_book_296.txt",
        "volume47_book_392.txt",
        "volume47_book_157.txt",
        "volume92_book_463.txt",
        "volume47_book_149.txt",
        "volume46_book_107.txt",
        "volume44_book_512.txt",
        "volume45_book_181.txt",
        "volume46_book_359.txt",
        "volume47_book_335.txt",
        "volume45_book_116.txt",
        "volume46_book_123.txt",
        "volume47_book_56.txt",
        "volume46_book_436.txt",
        "volume47_book_12.txt",
        "volume47_book_271.txt",
        "volume46_book_274.txt",
        "volume45_book_93.txt",
        "volume46_book_208.txt",
        "volume44_book_489.txt",
        "volume47_book_44.txt",
        "volume47_book_310.txt",
        "volume46_book_433.txt",
        "volume92_book_460.txt",
        "volume47_book_57.txt",
        "volume45_book_315.txt",
        "volume92_book_456.txt",
        "volume47_book_353.txt",
        "volume47_book_438.txt",
        "volume46_book_322.txt",
        "volume47_book_354.txt",
        "volume46_book_242.txt",
        "volume46_book_394.txt",
        "volume46_book_225.txt",
        "volume46_book_339.txt",
        "volume47_book_274.txt",
        "volume44_book_459.txt",
        "volume46_book_264.txt",
        "volume46_book_336.txt",
        "volume47_book_304.txt",
        "volume44_book_506.txt",
        "volume46_book_363.txt",
        "volume45_book_95.txt",
        "volume47_book_46.txt",
        "volume44_book_456.txt",
        "volume45_book_45.txt",
        "volume47_book_299.txt",
        "volume45_book_452.txt",
        "volume45_book_107.txt",
        "volume96_book_296.txt",
        "volume45_book_90.txt",
        "volume45_book_294.txt",
        "volume46_book_410.txt",
        "volume97_book_178.txt",
        "volume46_book_443.txt",
        "volume44_book_464.txt",
        "volume46_book_319.txt",
        "volume92_book_420.txt",
        "volume46_book_397.txt",
        "volume44_book_446.txt",
        "volume45_book_296.txt",
        "volume44_book_525.txt",
        "volume45_book_288.txt",
        "volume47_book_333.txt",
        "volume45_book_75.txt",
        "volume92_book_470.txt",
        "volume45_book_126.txt",
        "volume46_book_162.txt",
        "volume47_book_38.txt",
        "volume45_book_131.txt",
        "volume45_book_99.txt",
        "volume92_book_397.txt",
        "volume46_book_418.txt",
        "volume45_book_144.txt",
        "volume47_book_215.txt",
        "volume45_book_185.txt",
        "volume44_book_495.txt",
        "volume45_book_168.txt",
        "volume46_book_18.txt",
        "volume47_book_233.txt",
        "volume46_book_306.txt",
        "volume45_book_201.txt",
        "volume47_book_126.txt",
        "volume47_book_344.txt",
        "volume45_book_164.txt",
        "volume45_book_256.txt",
        "volume45_book_172.txt",
        "volume45_book_445.txt",
        "volume45_book_71.txt",
        "volume46_book_290.txt",
        "volume46_book_21.txt",
        "volume45_book_147.txt",
        "volume45_book_173.txt",
        "volume46_book_169.txt",
        "volume45_book_110.txt",
        "volume47_book_324.txt",
        "volume44_book_494.txt",
        "volume46_book_440.txt",
        "volume47_book_138.txt",
        "volume44_book_439.txt",
        "volume45_book_435.txt",
        "volume46_book_81.txt",
        "volume45_book_395.txt",
        "volume97_book_176.txt",
        "volume46_book_221.txt",
        "volume47_book_390.txt",
        "volume46_book_369.txt",
        "volume46_book_184.txt",
        "volume46_book_486.txt",
        "volume47_book_34.txt",
        "volume92_book_414.txt",
        "volume47_book_270.txt",
        "volume92_book_465.txt",
        "volume47_book_289.txt",
        "volume47_book_132.txt",
        "volume92_book_433.txt",
        "volume45_book_383.txt",
        "volume97_book_175.txt",
        "volume92_book_381.txt",
        "volume92_book_385.txt",
        "volume92_book_440.txt",
        "volume45_book_176.txt",
        "volume46_book_464.txt",
        "volume45_book_15.txt",
        "volume45_book_424.txt",
        "volume45_book_49.txt",
        "volume46_book_203.txt",
        "volume46_book_378.txt",
        "volume44_book_453.txt",
        "volume46_book_275.txt",
        "volume92_book_438.txt",
        "volume46_book_395.txt",
        "volume46_book_108.txt",
        "volume45_book_94.txt",
        "volume44_book_509.txt",
        "volume46_book_124.txt",
        "volume44_book_500.txt",
        "volume45_book_216.txt",
        "volume97_book_187.txt",
        "volume46_book_258.txt",
        "volume46_book_411.txt",
        "volume45_book_61.txt",
        "volume46_book_62.txt",
        "volume46_book_196.txt",
        "volume46_book_77.txt",
        "volume46_book_233.txt",
        "volume45_book_152.txt",
        "volume45_book_113.txt",
        "volume47_book_348.txt",
        "volume47_book_110.txt",
        "volume46_book_109.txt",
        "volume47_book_48.txt",
        "volume45_book_148.txt",
        "volume46_book_311.txt",
        "volume45_book_65.txt",
        "volume92_book_524.txt",
        "volume46_book_317.txt",
        "volume45_book_139.txt", "volume92_book_462.txt", "volume44_book_474.txt", "volume44_book_462.txt", "volume46_book_190.txt", "volume46_book_441.txt", "volume45_book_285.txt", "volume46_book_370.txt", "volume44_book_468.txt", "volume44_book_482.txt", "volume45_book_81.txt", "volume47_book_393.txt", "volume45_book_167.txt", "volume45_book_228.txt", "volume46_book_475.txt", "volume46_book_289.txt", "volume47_book_432.txt", "volume46_book_438.txt", "volume46_book_38.txt", "volume46_book_85.txt", "volume47_book_383.txt", "volume97_book_189.txt", "volume45_book_223.txt", "volume45_book_136.txt", "volume45_book_305.txt", "volume92_book_395.txt", "volume45_book_204.txt", "volume46_book_415.txt", "volume47_book_98.txt", "volume46_book_347.txt", "volume47_book_266.txt", "volume46_book_364.txt", "volume92_book_444.txt", "volume97_book_185.txt", "volume46_book_404.txt", "volume46_book_266.txt", "volume47_book_387.txt", "volume47_book_254.txt", "volume45_book_124.txt", "volume92_book_390.txt", "volume46_book_160.txt", "volume45_book_207.txt", "volume46_book_323.txt", "volume45_book_40.txt", "volume47_book_346.txt", "volume92_book_413.txt", "volume45_book_297.txt", "volume45_book_27.txt", "volume46_book_222.txt", "volume45_book_89.txt", "volume46_book_235.txt", "volume47_book_207.txt", "volume44_book_535.txt", "volume45_book_52.txt", "volume45_book_33.txt", "volume45_book_37.txt", "volume46_book_472.txt", "volume46_book_223.txt", "volume44_book_460.txt", "volume44_book_477.txt", "volume44_book_436.txt", "volume46_book_243.txt", "volume47_book_433.txt", "volume45_book_158.txt", "volume46_book_432.txt", "volume44_book_454.txt", "volume44_book_502.txt", "volume47_book_255.txt", "volume47_book_21.txt", "volume46_book_19.txt", "volume47_book_351.txt", "volume92_book_429.txt", "volume47_book_183.txt", "volume46_book_199.txt", "volume47_book_8.txt", "volume45_book_133.txt", "volume45_book_130.txt", "volume45_book_237.txt", "volume46_book_374.txt", "volume47_book_347.txt", "volume46_book_476.txt", "volume46_book_480.txt", "volume47_book_125.txt", "volume47_book_224.txt", "volume45_book_18.txt", "volume46_book_149.txt", "volume47_book_80.txt", "volume45_book_347.txt", "volume46_book_145.txt", "volume46_book_50.txt", "volume45_book_232.txt", "volume45_book_200.txt", "volume47_book_355.txt", "volume45_book_97.txt", "volume97_book_179.txt", "volume92_book_417.txt", "volume46_book_362.txt", "volume47_book_242.txt", "volume46_book_20.txt", "volume92_book_389.txt", "volume46_book_171.txt", "volume45_book_333.txt", "volume47_book_323.txt", "volume45_book_92.txt", "volume47_book_19.txt", "volume46_book_272.txt", "volume45_book_120.txt","volume46_book_447.txt", "volume45_book_153.txt", "volume46_book_344.txt", "volume44_book_521.txt", "volume45_book_6.txt", "volume46_book_361.txt", "volume46_book_133.txt", "volume45_book_60.txt", "volume46_book_251.txt", "volume46_book_423.txt", "volume46_book_117.txt", "volume47_book_131.txt", "volume92_book_401.txt", "volume44_book_458.txt", "volume46_book_172.txt", "volume47_book_403.txt", "volume46_book_187.txt", "volume45_book_447.txt", "volume46_book_444.txt", "volume46_book_380.txt", "volume46_book_318.txt", "volume46_book_137.txt", "volume44_book_485.txt", "volume46_book_485.txt", "volume46_book_42.txt", "volume92_book_400.txt", "volume45_book_20.txt", "volume46_book_439.txt", "volume97_book_177.txt", "volume44_book_508.txt", "volume46_book_419.txt", "volume44_book_490.txt", "volume47_book_243.txt", "volume47_book_272.txt", "volume47_book_250.txt", "volume47_book_340.txt", "volume46_book_304.txt", "volume46_book_249.txt", "volume46_book_63.txt", "volume47_book_162.txt", "volume47_book_155.txt", "volume46_book_72.txt", "volume46_book_237.txt", "volume47_book_61.txt", "volume47_book_408.txt", "volume45_book_106.txt", "volume92_book_426.txt", "volume47_book_267.txt", "volume45_book_247.txt", "volume46_book_87.txt", "volume47_book_221.txt", "volume45_book_206.txt", "volume46_book_86.txt", "volume46_book_218.txt", "volume47_book_187.txt", "volume44_book_534.txt", "volume44_book_493.txt", "volume47_book_225.txt", "volume46_book_92.txt", "volume47_book_100.txt", "volume47_book_343.txt", "volume47_book_223.txt", "volume46_book_408.txt", "volume97_book_182.txt", "volume46_book_101.txt", "volume47_book_371.txt", "volume46_book_492.txt", "volume45_book_44.txt", "volume47_book_14.txt", "volume47_book_386.txt", "volume47_book_87.txt", "volume46_book_106.txt", "volume46_book_192.txt", "volume46_book_265.txt", "volume47_book_227.txt", "volume47_book_417.txt", "volume45_book_346.txt", "volume45_book_248.txt", "volume45_book_304.txt", "volume46_book_393.txt", "volume45_book_295.txt", "volume46_book_204.txt", "volume46_book_379.txt", "volume46_book_357.txt", "volume46_book_365.txt", "volume47_book_359.txt", "volume47_book_318.txt", "volume47_book_102.txt", "volume46_book_224.txt", "volume46_book_41.txt", "volume45_book_209.txt", "volume45_book_453.txt", "volume45_book_437.txt", "volume46_book_343.txt", "volume47_book_50.txt", "volume46_book_315.txt", "volume46_book_292.txt", "volume46_book_146.txt", "volume47_book_337.txt", "volume46_book_396.txt", "volume46_book_189.txt", "volume45_book_132.txt", "volume92_book_404.txt", "volume45_book_199.txt", "volume45_book_197.txt", "volume45_book_41.txt", "volume47_book_42.txt", "volume45_book_32.txt", "volume46_book_489.txt", "volume46_book_442.txt", "volume44_book_519.txt", "volume46_book_313.txt", "volume92_book_461.txt", "volume46_book_297.txt", "volume46_book_80.txt", "volume46_book_375.txt", "volume45_book_448.txt", "volume46_book_128.txt", "volume44_book_428.txt", "volume97_book_184.txt", "volume47_book_349.txt", "volume46_book_417.txt", "volume46_book_91.txt", "volume47_book_374.txt", "volume92_book_424.txt", "volume47_book_39.txt", "volume45_book_35.txt", "volume47_book_238.txt", "volume47_book_397.txt", "volume44_book_523.txt", "volume46_book_496.txt", "volume46_book_39.txt", "volume92_book_382.txt", "volume46_book_294.txt", "volume47_book_31.txt", "volume47_book_91.txt", "volume45_book_17.txt", "volume45_book_231.txt", "volume47_book_206.txt", "volume46_book_70.txt", "volume46_book_84.txt", "volume47_book_24.txt", "volume46_book_352.txt", "volume45_book_221.txt", "volume44_book_470.txt", "volume47_book_406.txt", "volume45_book_151.txt", "volume45_book_390.txt", "volume47_book_202.txt", "volume47_book_54.txt", "volume44_book_529.txt", "volume46_book_465.txt", "volume46_book_324.txt", "volume45_book_335.txt", "volume44_book_505.txt", "volume44_book_486.txt", "volume45_book_203.txt", "volume47_book_53.txt", "volume45_book_84.txt", "volume46_book_299.txt", "volume46_book_76.txt", "volume46_book_99.txt", "volume46_book_168.txt", "volume45_book_100.txt", "volume92_book_416.txt", "volume46_book_334.txt", "volume92_book_454.txt", "volume46_book_469.txt", "volume47_book_237.txt", "volume47_book_27.txt", "volume47_book_244.txt", "volume45_book_68.txt", "volume46_book_291.txt", "volume44_book_449.txt", "volume47_book_435.txt", "volume46_book_67.txt", "volume47_book_434.txt", "volume46_book_355.txt", "volume45_book_10.txt", "volume47_book_439.txt", "volume46_book_4.txt", "volume47_book_437.txt", "volume47_book_29.txt", "volume45_book_169.txt", "volume46_book_358.txt", "volume47_book_15.txt", "volume47_book_234.txt", "volume92_book_439.txt", "volume97_book_186.txt", "volume45_book_382.txt", "volume47_book_103.txt", "volume46_book_202.txt", "volume44_book_467.txt", "volume45_book_2.txt", "volume47_book_298.txt", "volume46_book_121.txt", "volume47_book_66.txt", "volume46_book_437.txt", "volume47_book_136.txt", "volume95_book_108.txt", "volume44_book_492.txt", "volume45_book_4.txt", "volume47_book_391.txt", "volume44_book_480.txt", "volume46_book_32.txt", "volume44_book_423.txt", "volume46_book_327.txt", "volume46_book_353.txt", "volume46_book_252.txt", "volume46_book_373.txt", "volume46_book_491.txt", "volume46_book_94.txt", "volume45_book_5.txt", "volume46_book_48.txt", "volume46_book_182.txt", "volume92_book_442.txt", "volume45_book_76.txt", "volume46_book_178.txt", "volume46_book_314.txt", "volume45_book_146.txt", "volume45_book_214.txt", "volume47_book_11.txt", "volume45_book_165.txt", "volume47_book_189.txt", "volume45_book_322.txt", "volume92_book_446.txt", "volume47_book_292.txt", "volume44_book_463.txt", "volume47_book_59.txt", "volume92_book_412.txt", "volume45_book_155.txt", "volume47_book_22.txt", "volume47_book_184.txt", "volume46_book_345.txt", "volume46_book_434.txt", "volume92_book_387.txt", "volume45_book_16.txt", "volume47_book_400.txt", "volume45_book_193.txt", "volume44_book_520.txt", "volume46_book_159.txt", "volume46_book_12.txt", "volume46_book_474.txt", "volume47_book_235.txt", "volume47_book_67.txt", "volume45_book_313.txt", "volume47_book_385.txt", "volume47_book_245.txt", "volume46_book_141.txt", "volume44_book_450.txt", "volume47_book_33.txt", "volume47_book_336.txt", "volume47_book_246.txt", "volume47_book_297.txt", "volume46_book_268.txt", "volume92_book_457.txt", "volume47_book_362.txt", "volume45_book_11.txt", "volume92_book_406.txt", "volume47_book_220.txt", "volume47_book_376.txt", "volume44_book_507.txt", "volume45_book_22.txt", "volume46_book_303.txt", "volume46_book_8.txt", "volume46_book_377.txt", "volume47_book_192.txt", "volume45_book_109.txt", "volume46_book_337.txt", "volume47_book_65.txt", "volume46_book_105.txt", "volume45_book_64.txt", "volume46_book_367.txt", "volume46_book_366.txt", "volume46_book_340.txt", "volume46_book_494.txt", "volume46_book_127.txt", "volume47_book_311.txt", "volume92_book_409.txt", "volume46_book_406.txt", "volume46_book_44.txt", "volume46_book_240.txt", "volume46_book_257.txt", "volume45_book_170.txt", "volume47_book_239.txt", "volume46_book_181.txt", "volume46_book_356.txt", "volume45_book_215.txt", "volume45_book_7.txt", "volume46_book_177.txt", "volume45_book_192.txt", "volume47_book_194.txt", "volume44_book_441.txt", "volume45_book_154.txt", "volume46_book_321.txt", "volume97_book_180.txt", "volume46_book_405.txt", "volume46_book_421.txt", "volume45_book_25.txt",
        "volume45_book_42.txt", "volume46_book_134.txt", "volume47_book_135.txt", "volume47_book_217.txt", "volume46_book_176.txt", "volume44_book_479.txt", "volume46_book_422.txt", "volume45_book_105.txt", "volume47_book_360.txt", "volume46_book_209.txt", "volume47_book_363.txt", "volume46_book_446.txt", "volume44_book_471.txt", "volume47_book_203.txt", "volume45_book_26.txt", "volume46_book_245.txt", "volume47_book_147.txt", "volume47_book_124.txt", "volume44_book_435.txt", "volume47_book_158.txt", "volume47_book_420.txt", "volume46_book_256.txt", "volume45_book_3.txt", "volume45_book_122.txt", "volume45_book_159.txt", "volume45_book_443.txt", "volume45_book_74.txt", "volume44_book_513.txt", "volume47_book_253.txt", "volume46_book_206.txt", "volume46_book_413.txt", "volume44_book_501.txt", "volume46_book_445.txt", "volume45_book_293.txt", "volume47_book_30.txt", "volume45_book_145.txt", "volume46_book_188.txt", "volume47_book_10.txt", "volume46_book_270.txt", "volume92_book_452.txt", "volume46_book_255.txt", "volume45_book_13.txt", "volume47_book_37.txt", "volume46_book_407.txt", "volume46_book_118.txt", "volume46_book_219.txt", "volume45_book_108.txt", "volume46_book_241.txt", "volume45_book_198.txt", "volume44_book_498.txt", "volume47_book_63.txt", "volume47_book_18.txt", "volume44_book_476.txt", "volume45_book_111.txt", "volume46_book_335.txt", "volume92_book_403.txt", "volume45_book_118.txt", "volume45_book_39.txt", "volume46_book_269.txt", "volume47_book_193.txt", "volume46_book_338.txt", "volume46_book_280.txt", "volume46_book_53.txt", "volume92_book_410.txt", "volume47_book_190.txt", "volume45_book_137.txt", "volume45_book_163.txt", "volume46_book_194.txt", "volume92_book_447.txt", "volume47_book_401.txt", "volume45_book_196.txt", "volume46_book_10.txt", "volume46_book_262.txt", "volume44_book_426.txt", "volume46_book_301.txt", "volume46_book_467.txt", "volume45_book_31.txt", "volume46_book_300.txt", "volume47_book_123.txt", "volume45_book_171.txt", "volume47_book_373.txt", "volume44_book_466.txt", "volume47_book_94.txt", "volume46_book_147.txt", "volume45_book_21.txt", "volume44_book_524.txt", "volume46_book_479.txt", "volume92_book_428.txt", "volume47_book_372.txt", "volume47_book_278.txt", "volume46_book_267.txt", "volume45_book_243.txt", "volume47_book_156.txt", "volume46_book_471.txt", "volume46_book_69.txt", "volume45_book_245.txt", "volume46_book_135.txt", "volume45_book_202.txt", "volume46_book_248.txt", "volume45_book_51.txt", "volume47_book_36.txt", "volume46_book_326.txt", "volume46_book_253.txt", "volume44_book_438.txt", "volume45_book_442.txt", "volume46_book_75.txt", "volume45_book_79.txt", "volume45_book_73.txt", "volume46_book_59.txt", "volume47_book_256.txt", "volume92_book_422.txt", "volume46_book_132.txt", "volume92_book_458.txt", "volume45_book_67.txt", "volume45_book_255.txt", "volume44_book_497.txt", "volume47_book_240.txt", "volume47_book_440.txt", "volume46_book_170.txt", "volume46_book_45.txt", "volume46_book_205.txt", "volume92_book_423.txt", "volume45_book_69.txt", "volume44_book_425.txt", "volume45_book_80.txt", "volume47_book_96.txt", "volume47_book_219.txt", "volume92_book_445.txt", "volume46_book_263.txt", "volume46_book_302.txt", "volume47_book_196.txt", "volume46_book_239.txt", "volume47_book_241.txt", "volume44_book_533.txt", "volume44_book_430.txt", "volume47_book_40.txt", "volume45_book_227.txt", "volume47_book_312.txt", "volume45_book_29.txt", "volume46_book_93.txt", "volume46_book_420.txt", "volume46_book_83.txt", "volume46_book_119.txt", "volume92_book_407.txt", "volume46_book_261.txt", "volume46_book_431.txt", "volume47_book_45.txt", "volume92_book_431.txt", "volume44_book_475.txt", "volume46_book_376.txt", "volume44_book_530.txt", "volume45_book_102.txt", "volume44_book_469.txt", "volume47_book_173.txt", "volume45_book_140.txt", "volume47_book_389.txt", "volume47_book_195.txt", "volume46_book_260.txt", "volume46_book_11.txt", "volume46_book_298.txt", "volume92_book_388.txt", "volume46_book_207.txt", "volume46_book_49.txt", "volume44_book_431.txt", "volume47_book_436.txt", "volume45_book_24.txt", "volume46_book_351.txt", "volume44_book_424.txt", "volume47_book_55.txt", "volume46_book_473.txt", "volume46_book_103.txt", "volume92_book_421.txt", "volume47_book_60.txt", "volume45_book_121.txt", "volume47_book_86.txt", "volume47_book_228.txt", "volume45_book_78.txt", "volume45_book_103.txt", "volume92_book_419.txt", "volume47_book_361.txt", "volume47_book_334.txt", "volume46_book_403.txt", "volume92_book_467.txt", "volume47_book_236.txt", "volume47_book_429.txt", "volume46_book_463.txt", "volume46_book_342.txt", "volume44_book_455.txt", "volume45_book_160.txt", "volume46_book_244.txt", "volume92_book_399.txt", "volume44_book_451.txt", "volume45_book_66.txt", "volume46_book_68.txt", "volume46_book_325.txt", "volume46_book_66.txt", "volume47_book_35.txt", "volume47_book_291.txt", "volume92_book_441.txt", "volume45_book_166.txt", "volume47_book_99.txt", "volume46_book_478.txt", "volume45_book_244.txt", "volume46_book_468.txt", "volume47_book_232.txt", "volume45_book_259.txt", "volume45_book_343.txt", "volume45_book_269.txt", "volume45_book_119.txt", "volume45_book_149.txt", "volume47_book_364.txt", "volume92_book_436.txt", "volume44_book_510.txt", "volume45_book_28.txt", "volume47_book_146.txt", "volume47_book_84.txt", "volume44_book_437.txt", "volume46_book_349.txt", "volume44_book_444.txt", "volume47_book_161.txt", "volume46_book_193.txt", "volume46_book_409.txt", "volume45_book_402.txt", "volume46_book_481.txt", "volume44_book_445.txt", "volume44_book_488.txt", "volume46_book_247.txt", "volume44_book_504.txt", "volume47_book_101.txt", "volume44_book_511.txt", "volume45_book_104.txt", "volume47_book_407.txt", "volume45_book_98.txt", "volume45_book_238.txt", "volume92_book_396.txt", "volume46_book_276.txt", "volume45_book_30.txt", "volume45_book_446.txt", "volume45_book_14.txt", "volume45_book_450.txt", "volume46_book_259.txt", "volume45_book_190.txt", "volume46_book_320.txt", "volume92_book_383.txt", "volume47_book_375.txt", "volume92_book_464.txt", "volume44_book_515.txt", "volume47_book_20.txt", "volume46_book_201.txt", "volume46_book_15.txt", "volume45_book_127.txt", "volume45_book_381.txt", "volume45_book_239.txt", "volume46_book_200.txt", "volume44_book_465.txt", "volume44_book_487.txt", "volume47_book_288.txt", "volume92_book_402.txt", "volume47_book_95.txt", "volume46_book_52.txt", "volume47_book_118.txt", "volume46_book_148.txt", "volume44_book_528.txt", "volume47_book_127.txt", "volume45_book_143.txt", "volume44_book_532.txt", "volume44_book_434.txt", "volume47_book_431.txt", "volume47_book_93.txt", "volume47_book_13.txt", "volume45_book_129.txt", "volume97_book_174.txt", "volume47_book_117.txt", "volume92_book_415.txt", "volume92_book_386.txt", "volume46_book_487.txt", "volume45_book_59.txt", "volume45_book_63.txt", "volume44_book_461.txt", "volume47_book_16.txt", "volume45_book_162.txt", "volume46_book_271.txt", "volume47_book_9.txt", "volume47_book_252.txt", "volume45_book_12.txt", "volume92_book_435.txt", "volume92_book_448.txt", "volume46_book_246.txt", "volume44_book_447.txt", "volume45_book_426.txt", "volume44_book_496.txt", "volume45_book_70.txt"
    ])

    # --- Find all .txt files in the input directory ---
    search_path = os.path.join(args.input_dir, "*.txt")
    all_text_files = glob.glob(search_path)

    # --- Filter only allowed files ---
    text_files = [f for f in all_text_files if os.path.basename(f).strip().lower() in allowed_files]

    # --- Debug info ---
    print("Total files in folder:", len(all_text_files))
    print("Total allowed files:", len(allowed_files))
    print("Total matching files:", len(text_files))
    print("Example matches:", [os.path.basename(f) for f in text_files[:10]])

    # --- Process files and save to JSONL ---
    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        for file_path in tqdm(text_files, desc="Processing files"):
            doc_id = os.path.splitext(os.path.basename(file_path))[0]
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                processed_data = process_document(doc_id, content, nlp)
                if processed_data:
                    outfile.write(json.dumps(processed_data) + '\n')
            except Exception as e:
                print(f"Failed to read or process file {file_path}: {e}")

    print(f"\n✅ Processing complete. Output saved to '{args.output_file}'.")

if __name__ == "__main__":
    main()
