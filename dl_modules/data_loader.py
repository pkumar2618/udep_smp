
def generate_batches(dataset, batch_size, shuffle=True, drop_last=True, device="cpu"):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_data_dict = {}

        for name, tensor in data_dict.items():
            out_data_dict[name] = data_dict[name].to(device)

        yield out_data_dict