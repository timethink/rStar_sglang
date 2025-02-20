MODEL="/mnt/teamdrive/xy/xy/sft/0219/test_rstar7b_jump/global_step_20/actor-huggingface"

# truncate -s 0 "$MODEL"/amc23.jsonl
# python eval.py --model "$MODEL" --device 0 --task amc23 
# python eval_output.py --file_path $MODEL"/amc23.jsonl" >> /home/aiscuser/rStar/test_output/amc23.txt

truncate -s 0 "$MODEL"/aime2024.jsonl
python eval.py --model "$MODEL" --device 0 --task aime2024
python eval_output.py --file_path $MODEL"/aime2024.jsonl" >> /home/aiscuser/rStar/test_output/aime2024.txt

# truncate -s 0 "$MODEL"/gaokao2023en.jsonl
# python eval.py --model "$MODEL" --device 0 --task gaokao2023en
# python eval_output.py --file_path $MODEL"/gaokao2023en.jsonl" >> /home/aiscuser/rStar/test_output/gaokao2023en.txt

# truncate -s 0 "$MODEL"/gsm8k.jsonl
# python eval.py --model "$MODEL" --device 0 --task gsm8k
# python eval_output.py --file_path $MODEL"/gsm8k.jsonl" >> /home/aiscuser/rStar/test_output/gsm8k.txt

truncate -s 0 "$MODEL"/math500.jsonl
python eval.py --model "$MODEL" --device 0 --task math500
python eval_output.py --file_path $MODEL"/math500.jsonl" >> /home/aiscuser/rStar/test_output/math.txt

# truncate -s 0 "$MODEL"/collegemath.jsonl
# python eval.py --model "$MODEL" --device 0 --task collegemath
# python eval_output.py --file_path $MODEL"/collegemath.jsonl" >> /home/aiscuser/rStar/test_output/collegemath.txt