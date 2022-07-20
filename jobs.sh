# S3DIS
python main_S3DIS.py --mode train --gpu 0 --test_area 5 --labeled_point 0.1%
python main_S3DIS.py --mode test --gpu 0 --test_area 5

python main_S3DIS.py --mode test --gpu 0 --test_area 1 --gen_pesudo
python main_S3DIS.py --mode test --gpu 0 --test_area 2 --gen_pesudo
python main_S3DIS.py --mode test --gpu 0 --test_area 3 --gen_pesudo
python main_S3DIS.py --mode test --gpu 0 --test_area 4 --gen_pesudo
python main_S3DIS.py --mode test --gpu 0 --test_area 6 --gen_pesudo
python main_S3DIS.py --mode train --gpu 0 --test_area 5 --labeled_point 0.1% --retrain
python main_S3DIS.py --mode test --gpu 0 --test_area 5 --labeled_point 0.1%


# Toronto3D
python main_Toronto3D.py --mode train --gpu 0 --test_area 2 --labeled_point 0.1%
python main_Toronto3D.py --mode test --gpu 0 --test_area 2 --labeled_point 0.1%

# Semantic3D
python main_Semantic3D.py --mode train --gpu 0 --labeled_point 0.1%
python main_Semantic3D.py --mode test --gpu 0 --labeled_point 0.1%

# SemanticKITTI
python main_SemanticKITTI.py --mode train --gpu 0 --labeled_point 0.1%
python main_SemanticKITTI.py --mode test --gpu 0 --test_area 08

python -B main_SemanticKITTI.py --gpu 0 --mode test --test_area 11
python -B main_SemanticKITTI.py --gpu 0 --mode test --test_area 12
python -B main_SemanticKITTI.py --gpu 0 --mode test --test_area 13
python -B main_SemanticKITTI.py --gpu 0 --mode test --test_area 14
python -B main_SemanticKITTI.py --gpu 0 --mode test --test_area 15
python -B main_SemanticKITTI.py --gpu 0 --mode test --test_area 16
python -B main_SemanticKITTI.py --gpu 0 --mode test --test_area 17
python -B main_SemanticKITTI.py --gpu 0 --mode test --test_area 18
python -B main_SemanticKITTI.py --gpu 0 --mode test --test_area 19
python -B main_SemanticKITTI.py --gpu 0 --mode test --test_area 20
python -B main_SemanticKITTI.py --gpu 0 --mode test --test_area 21
