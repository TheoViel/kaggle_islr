import torch


def compute_adjacency_features(x, embed):
    bs, n_frames, n_landmarks, n_fts = x.size()
    embed = (
        embed[:, 0].unsqueeze(1).repeat(1, n_frames, 1).view(-1)
    )  # this avoids padding
    left_hand = x.view(-1, n_fts)[embed == 5].view(bs, n_frames, -1, n_fts)
    adj_left_hand = (
        ((left_hand.unsqueeze(-2) - left_hand.unsqueeze(-3)) ** 2).sum(-1).sqrt()
    )

    right_hand = x.view(-1, n_fts)[embed == 10].view(bs, n_frames, -1, n_fts)
    adj_right_hand = (
        ((right_hand.unsqueeze(-2) - right_hand.unsqueeze(-3)) ** 2).sum(-1).sqrt()
    )

    return torch.cat(
        [
            adj_left_hand.view(bs, n_frames, -1),
            adj_right_hand.view(bs, n_frames, -1),
        ],
        -1,
    )


def compute_adjacency_matrix(x_pos):
    mask = (x_pos == 0).sum(-1) != 3
    adj = (
        ((x_pos.unsqueeze(-2) - x_pos.unsqueeze(-3)) ** 2).sum(-1).sqrt()
    )  # distance matrix
    adj = 1 / (adj + 1)
    adj = adj * mask.unsqueeze(-1) * mask.unsqueeze(-2)

    adj = torch.stack(
        [
            adj,
            torch.ones_like(adj),
            (adj > 0.5).float(),
            (adj > 0.75).float(),
            (adj > 0.9).float(),
        ],
        2,
    )
    return adj


def compute_hand_features(x, embed):
    bs, n_frames, n_landmarks, n_fts = x.size()
    embed = (
        embed[:, 0].unsqueeze(1).repeat(1, n_frames, 1).view(-1)
    )  # this avoids padding
    left_hand = x.view(-1, n_fts)[embed == 5].view(bs, n_frames, -1, n_fts)
    adj_left_hand = (
        ((left_hand.unsqueeze(-2) - left_hand.unsqueeze(-3)) ** 2).sum(-1).sqrt()
    )

    sz = adj_left_hand.size(3)
    adj_left_hand = adj_left_hand.view(bs * n_frames, sz, -1)
    ids_a, ids_b = torch.triu_indices(sz, sz, offset=1).unbind()
    adj_left_hand = adj_left_hand[:, ids_a, ids_b].view(bs, n_frames, -1)

    right_hand = x.view(-1, n_fts)[embed == 10].view(bs, n_frames, -1, n_fts)
    adj_right_hand = (
        ((right_hand.unsqueeze(-2) - right_hand.unsqueeze(-3)) ** 2).sum(-1).sqrt()
    )

    sz = adj_right_hand.size(3)
    adj_right_hand = adj_right_hand.view(bs * n_frames, sz, -1)
    ids_a, ids_b = torch.triu_indices(sz, sz, offset=1).unbind()
    adj_right_hand = adj_right_hand[:, ids_a, ids_b].view(bs, n_frames, -1)

    return torch.cat([adj_left_hand, adj_right_hand], -1)


def compute_hand_to_face_distances(x, embed):
    bs, n_frames, n_landmarks, n_fts = x.size()
    embed = (
        embed[:, 0].unsqueeze(1).repeat(1, n_frames, 1).view(-1)
    )  # this avoids padding

    face = x.view(-1, n_fts)[
        torch.isin(embed, torch.tensor([3, 4, 8, 9, 7]).to(x.device))
    ].view(bs, n_frames, -1, n_fts)

    left_hand = x.view(-1, n_fts)[embed == 5].view(bs, n_frames, -1, n_fts)
    right_hand = x.view(-1, n_fts)[embed == 10].view(bs, n_frames, -1, n_fts)

    left_dists = (
        ((left_hand.unsqueeze(-2) - face.unsqueeze(-3)) ** 2).sum(-1).sqrt()
    )
    right_dists = (
        ((right_hand.unsqueeze(-2) - face.unsqueeze(-3)) ** 2).sum(-1).sqrt()
    )

    return torch.cat([left_dists, right_dists], -1).view(bs, n_frames, -1)


def add_shift(x, n=1):
    padding = torch.zeros((x.size(0), n, x.size(2), x.size(3)), device=x.device)
    x = torch.cat(
        [
            torch.cat([x[:, n:], padding], axis=1),
            x,
            torch.cat([padding, x[:, :-n]], axis=1),
        ],
        axis=3,
    )
    return x


def add_speed(x, n=1):
    padding = torch.zeros((x.size(0), n, x.size(2), x.size(3)), device=x.device)
    x = torch.cat(
        [
            torch.cat([x[:, n:], padding], axis=1) - x,
            x,
            x - torch.cat([padding, x[:, :-n]], axis=1),
        ],
        axis=3,
    )
    return x
