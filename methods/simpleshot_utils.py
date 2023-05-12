import numpy as np
from torch.autograd import Variable


def ss_episode(model, x, n_way, n_shot, n_query, out_mean):
    output, _ = ss_step(model, x, n_way, n_shot, n_query)
    output = output.view(n_way, n_shot + n_query, -1)
    support = output[:, :n_shot]
    query = output[:, n_shot:]

    support_meaned = support.contiguous().mean(1)
    query = query.contiguous().view(n_way * n_query, -1)

    pred, distance = metric_class_type(support_meaned.cpu().numpy(),
                                       query.cpu().numpy(),
                                       base_mean=out_mean,
                                       k=1)
    return pred, distance, (support, query)


def ss_step(model, x, n_way, n_shot, n_query):
    x = Variable(x.cuda())
    x = x.contiguous().view(n_way * (n_shot + n_query), *x.size()[2:])
    output, fc_output = model(x, True)
    return output, fc_output


def metric_class_type(support, query, base_mean, k=1):
    support -= base_mean
    support /= np.linalg.norm(support, 2, 1)[:, None]

    query -= base_mean
    query /= np.linalg.norm(query, 2, 1)[:, None]

    subtract = support[:, None, :] - query
    distance = np.linalg.norm(subtract, 2, axis=-1)

    idx = np.argpartition(distance, k, axis=0)[:k]
    return idx, distance
