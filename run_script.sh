python inference.py \
  --image_path "test/vase1.jpg" \
  --src_prompt "a vase of colorful flowers on a table." \
  --prompts "a vase of red roses on a table" \
            "a stainless steel vase of red roses on a table" \
            "a stainless steel vase of red roses on a table with an apple nearby" \
  --output_dir "results/test" \
  --seed 42

python inference.py \
  --image_path "test/couch.jpg" \
  --src_prompt "a couch with pillows sitting in front of a wall." \
  --prompts "a couch with pillows sitting in front of a forest." \
            "a couch with pillows sitting in front of a forest with red leaves." \
            "a leather couch with pillows sitting in front of a forest with red leaves." \
            "a red-brown leather couch with pillows sitting in front of a forest with red leaves" \
  --output_dir "results/test" \
  --seed 42

python inference.py \
  --image_path "test/woman.jpg" \
  --src_prompt "illustration of a woman meditating in a yoga pose." \
  --prompts "illustration of a woman meditating in a yoga pose in the background with moon." \
            "illustration of a woman meditating in a yoga pose in the background with moon, with a small candle right beside her." \
            "illustration of a woman wearing a pink sweater meditating in a yoga pose in the background with moon, with a small candle right beside her." \
            "illustration of a woman with a gentle smile wearing a pink sweater meditating in a yoga pose in the background with moon, with a small candle right beside her." \
  --output_dir "results/test" \
  --seed 42

python inference.py \
  --image_path "test/crow.jpg" \
  --src_prompt "A black and gray crow standing on the ground." \
  --prompts "A black and gray crow standing on the grass." \
            "A disney style of a black and gray crow standing on the grass." \
            "A disney style of a black and gray crow standing on the grass with some clovers." \
            "A disney style of a black and gray eagle standing on the grass with some clovers." \
            "A disney style of a black and gray eagle standing on the grass with some clovers and a small stone." \
  --output_dir "results/test" \
  --seed 42