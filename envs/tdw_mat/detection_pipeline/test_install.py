from mmdet.apis import init_detector, inference_detector

config_file = './detection_pipeline/config.py'
checkpoint_file = './detection_pipeline/epoch_1.pth'
model = init_detector(config_file, checkpoint_file, device='cpu')  # or device='cuda:0'

print(inference_detector(model, '/home/winnie/dwh/vision/Co-LLM-Agents/envs/tdw_mat/results/vision-single-run2/7/Images/0/0127_0607.png'))