import torch
import numpy as np



def replace_layers(model, i, indexes, layers):
    if i in indexes:
        return layers[indexes.index(i)]
    return model[i]
def prune_layer(model, layer_index, filter_index, conv_flag=True):
    use_cuda = torch.cuda.is_available()
    if conv_flag:
        conv = model.features[layer_index]
        next_conv = None
        offset = 1

        for res in model.features[layer_index + 1 : ]:
            if isinstance(res, torch.nn.modules.conv.Conv2d):
                next_conv = res
                break
            offset += 1
        new_conv = \
            torch.nn.Conv2d(in_channels = conv.in_channels, \
                out_channels = conv.out_channels - 1,
                kernel_size = conv.kernel_size, \
                stride = conv.stride,
                padding = conv.padding,
                dilation = conv.dilation,
                groups = conv.groups,
                bias = (conv.bias is not None))

        old_weights = conv.weight.data.cpu().numpy()
        new_weights = new_conv.weight.data.cpu().numpy()

        new_weights[: filter_index, :, :, :] = old_weights[: filter_index, :, :, :]
        new_weights[filter_index : , :, :, :] = old_weights[filter_index + 1 :, :, :, :]
        new_conv.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            new_conv.weight.data = new_conv.weight.data.cuda()
        bias_numpy = conv.bias.data.cpu().numpy()
        bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
        bias[:filter_index] = bias_numpy[:filter_index]
        bias[filter_index : ] = bias_numpy[filter_index + 1 :]
        new_conv.bias.data = torch.from_numpy(bias)
        if use_cuda:
            new_conv.bias.data = new_conv.bias.data.cuda()
        if not next_conv is None:
            next_new_conv = \
                torch.nn.Conv2d(in_channels = next_conv.in_channels - 1,\
                    out_channels =  next_conv.out_channels, \
                    kernel_size = next_conv.kernel_size, \
                    stride = next_conv.stride,
                    padding = next_conv.padding,
                    dilation = next_conv.dilation,
                    groups = next_conv.groups,
                    bias = (next_conv.bias is not None))

            old_weights = next_conv.weight.data.cpu().numpy()
            new_weights = next_new_conv.weight.data.cpu().numpy()

            new_weights[:, : filter_index, :, :] = old_weights[:, : filter_index, :, :]
            new_weights[:, filter_index : , :, :] = old_weights[:, filter_index + 1 :, :, :]
            next_new_conv.weight.data = torch.from_numpy(new_weights)
            if use_cuda:
                next_new_conv.weight.data = next_new_conv.weight.data.cuda()
            next_new_conv.bias.data = next_conv.bias.data
        if not next_conv is None:
            features = torch.nn.Sequential(
                    *(replace_layers(model.features, i, [layer_index, layer_index+offset], \
                        [new_conv, next_new_conv]) for i, _ in enumerate(model.features)))
            del model.features
            del conv

            model.features = features

        else:
            #Prunning the last conv layer. This affects the first linear layer of the classifier.
            model.features = torch.nn.Sequential(
                    *(replace_layers(model.features, i, [layer_index], \
                        [new_conv]) for i, _ in enumerate(model.features)))
            layer_index = 0
            old_linear_layer = None
            for module in model.classifier:
                if isinstance(module, torch.nn.Linear):
                    old_linear_layer = module
                    break
                layer_index = layer_index  + 1

            if old_linear_layer is None:
                raise BaseException("No linear laye found in classifier")
            params_per_input_channel = old_linear_layer.in_features // conv.out_channels

            new_linear_layer = \
                torch.nn.Linear(old_linear_layer.in_features - params_per_input_channel, 
                    old_linear_layer.out_features)

            old_weights = old_linear_layer.weight.data.cpu().numpy()
            new_weights = new_linear_layer.weight.data.cpu().numpy()        

            new_weights[:, : filter_index * params_per_input_channel] = \
                old_weights[:, : filter_index * params_per_input_channel]
            new_weights[:, filter_index * params_per_input_channel :] = \
                old_weights[:, (filter_index + 1) * params_per_input_channel :]

            new_linear_layer.bias.data = old_linear_layer.bias.data

            new_linear_layer.weight.data = torch.from_numpy(new_weights)
            if use_cuda:
                new_linear_layer.weight.data = new_linear_layer.weight.data.cuda()

            classifier = torch.nn.Sequential(
                *(replace_layers(model.classifier, i, [layer_index], \
                    [new_linear_layer]) for i, _ in enumerate(model.classifier)))

            del model.classifier
            del next_conv
            del conv
            model.classifier = classifier

        return model
    else:
        line = model.classifier[layer_index]
        next_line = None
        offset = 1
        for res in model.classifier[layer_index + 1 : ]:
            if isinstance(res, torch.nn.modules.linear.Linear):
                next_line = res
                break
            offset += 1
        if next_line is None:
            print('ERROR CANNOT PRUNE LAST LAYER')
            return model
        new_line = \
            torch.nn.Linear(in_features = line.in_features, \
                out_features = line.out_features - 1,
                bias = (line.bias is not None))

        old_weights = line.weight.data.cpu().numpy()
        new_weights = new_line.weight.data.cpu().numpy()

        new_weights[: filter_index, :] = old_weights[: filter_index,:]
        new_weights[filter_index : , :] = old_weights[filter_index + 1 :, :]
        new_line.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            new_line.weight.data = new_line.weight.data.cuda()

        bias_numpy = line.bias.data.cpu().numpy()

        bias = np.zeros(shape = (bias_numpy.shape[0] - 1), dtype = np.float32)
        bias[:filter_index] = bias_numpy[:filter_index]
        bias[filter_index : ] = bias_numpy[filter_index + 1 :]
        new_line.bias.data = torch.from_numpy(bias)
        if use_cuda:
            new_line.bias.data = new_line.bias.data.cuda()
        next_new_line = \
            torch.nn.Linear(in_features = next_line.in_features - 1, \
                out_features = next_line.out_features,
                bias = (next_line.bias is not None))

        old_weights = next_line.weight.data.cpu().numpy()
        new_weights = next_new_line.weight.data.cpu().numpy()

        new_weights[:, : filter_index] = old_weights[:, : filter_index]
        new_weights[:, filter_index :] = old_weights[:, filter_index + 1 :]
        next_new_line.weight.data = torch.from_numpy(new_weights)
        if use_cuda:
            next_new_line.weight.data = next_new_line.weight.data.cuda()
        next_new_line.bias.data = next_line.bias.data
        classifier = torch.nn.Sequential(
                *(replace_layers(model.classifier, i, [layer_index, layer_index+offset], \
                    [new_line, next_new_line]) for i, _ in enumerate(model.classifier)))
        del model.classifier
        del line

        model.classifier = classifier
        return model
