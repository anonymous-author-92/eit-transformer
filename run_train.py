from libs import *
from torch.utils.data import DataLoader

get_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ADD_GRAD_CHANNEL = True
N_CHANNEL = 1
PARTS = [3,4,5,6]

train_dataset = EITDataset(part_idx=PARTS,
                           file_type='h5',
                           subsample=1, # Unet baseline is using 101**2
                           channel=N_CHANNEL,
                           return_grad=ADD_GRAD_CHANNEL,
                           online_grad=False,
                           train_data=True,)
train_loader = DataLoader(train_dataset,
                          batch_size=10,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=True)

valid_dataset = EITDataset(part_idx=PARTS,
                           file_type='h5',
                           channel=N_CHANNEL,
                           subsample=1,
                           return_grad=ADD_GRAD_CHANNEL,
                           online_grad=False,
                           train_data=False,)
valid_loader = DataLoader(valid_dataset,
                          batch_size=20,
                          shuffle=False,
                          drop_last=False,
                          pin_memory=True)

config = dict(in_chan=N_CHANNEL*(1+2*ADD_GRAD_CHANNEL),
            base_chan=32,
            input_size=(256, 256),
            num_classes=1,
            reduce_size=16,
            num_blocks=[1, 1, 1, 1],
            num_heads=[4, 4, 4, 4],
            projection='interp',
            attn_drop=0.1,
            proj_drop=0.1,
            rel_pos=True,
            aux_loss=True,
            maxpool=True,
            bias=True,
            add_grad_channel=True,)
model = UTransformer(**config)
model.to(device)
print(get_num_params(model))

epochs = 50

lr = 1e-3
h = 1/201
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = OneCycleLR(optimizer, max_lr=lr, div_factor=1e3, final_div_factor=1e4,
                       steps_per_epoch=len(train_loader), pct_start=0.2, epochs=epochs)

loss_func = CrossEntropyLoss2d(regularizer=False,
                               h=h, gamma=0.1,
                               debug=False)
metric_func = L2Loss2d(regularizer=False)

result = run_train(model, loss_func, metric_func,
                   train_loader, valid_loader,
                   optimizer, scheduler,
                   train_batch=train_batch_eit,
                   validate_epoch=validate_epoch_eit,
                   epochs=epochs,
                   patience=None,
                   model_name=f'ut.pt',
                   result_name='eit-ut.pkl',
                   tqdm_mode='batch',
                   mode='min',
                   device=device)


model = UTransformer(**config)
model.load_state_dict(torch.load(os.path.join(MODEL_PATH, f'ut.pt')))
model.to(device)

valid_loader = DataLoader(valid_dataset,
                          batch_size=50,
                          shuffle=False,
                          drop_last=False,
                          pin_memory=True)
metric_func = L2Loss2d(regularizer=False, h=h)
val_result = validate_epoch_eit(model, metric_func, valid_loader, device)
print(val_result['metric'])
