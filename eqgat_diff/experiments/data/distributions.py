import torch
from torch.distributions.categorical import Categorical


def get_distributions(args, dataset_info, datamodule):
    histogram = dataset_info["n_nodes"]
    in_node_nf = len(dataset_info["atom_decoder"]) + int(args.include_charges)
    nodes_dist = DistributionNodes(histogram)

    prop_dist = None
    if len(args.properties_list) > 0:
        prop_dist = DistributionProperty(datamodule, args.properties_list)

    return nodes_dist, prop_dist


class DistributionNodes:
    def __init__(self, histogram):
        """Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
        historgram: dict. The keys are num_nodes, the values are counts
        """

        if type(histogram) == dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        probas = self.prob[batch_n_nodes.to(self.prob.device)]
        log_p = torch.log(probas + 1e-10)
        return log_p.to(batch_n_nodes.device)


PROP_TO_IDX_GEOMQM = {
    "dipole_norm": 0,
    "total_energy": 1,
    "HOMO-LUMO_gap": 2,
    "dispersion": 3,
    "atomisation_energy": 4,
    "polarizability": 5,
}

IDX_TO_PROP_GEOMQM = {v: k for k, v in PROP_TO_IDX_GEOMQM.items()}

PROP_TO_IDX_AQM = {
    "DIP": 0,
    "HLgap": 1,
    "eAT": 2,
    "eC": 3,
    "eEE": 4,
    "eH": 5,
    "eKIN": 6,
    "eKSE": 7,
    "eL": 8,
    "eNE": 9,
    "eNN": 10,
    "eMBD": 11,
    "eTS": 12,
    "eX": 13,
    "eXC": 14,
    "eXX": 15,
    "mPOL": 16,
}

IDX_TO_PROP_AQM = {v: k for k, v in PROP_TO_IDX_AQM.items()}

PROP_TO_IDX_AQM_QM7X = {
    "DIP": 0,
    "HLgap": 1,
    "eAT": 2,
    "eC": 3,
    "eEE": 4,
    "eH": 5,
    "eKIN": 6,
    "eL": 7,
    "eNE": 8,
    "eNN": 9,
    "eMBD": 10,
    "eTS": 11,
    "eX": 12,
    "eXC": 13,
    "eXX": 14,
    "mPOL": 15,
}

IDX_TO_PROP_AQM_QM7X = {v: k for k, v in PROP_TO_IDX_AQM_QM7X.items()}

PROP_TO_IDX_QM9 = {
    "dipole_moment": 0,
    "isotropic_polarizability": 1,
    "homo": 2,
    "lumo": 3,
    "gap": 4,
    "electronic_spatial_extent": 5,
    "zpve": 6,
    "energy_u0": 7,
    "energy_U": 8,
    "enthalpy_H": 9,
    "free_energy": 10,
    "heat_capacity": 11,
}

IDX_TO_PROP_QM9 = {v: k for k, v in PROP_TO_IDX_QM9.items()}


class DistributionProperty:
    def __init__(
        self,
        dataset,
        properties,
        num_bins=1000,
        normalizer=None,
    ):
        self.num_bins = num_bins
        self.distributions = {}
        self.properties = properties

        # train_idx = [int(i) for i in datamodule.idx_train]
        # mols = datamodule.dataset.data.mol
        # num_atoms = torch.tensor(
        #     [a.GetNumAtoms() for i, a in enumerate(mols) if i in train_idx]
        # )

        n_nodes_train = self.calculate_n_nodes(dataset.train_dataset)
        n_nodes_val = self.calculate_n_nodes(dataset.val_dataset)
        n_nodes_test = self.calculate_n_nodes(dataset.test_dataset)
        n_nodes = torch.cat([n_nodes_train, n_nodes_val, n_nodes_test])

        for prop in properties:
            self.distributions[prop] = {}

            # idx = datamodule.dataset.label2idx[prop]
            # property = datamodule.dataset.data.y[:, idx]
            # property = torch.tensor(
            #     [a for i, a in enumerate(property) if i in train_idx]
            # )
            idx = dataset.label2idx[prop]
            prop_train = dataset.train_dataset.data.y[:, idx]
            prop_val = dataset.val_dataset.data.y[:, idx]
            prop_test = dataset.test_dataset.data.y[:, idx]
            properties = torch.cat([prop_train, prop_val, prop_test], dim=0)
            self._create_prob_dist(n_nodes, properties, self.distributions[prop])

        self.normalizer = normalizer

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _create_prob_dist(self, nodes_arr, values, distribution):
        min_nodes, max_nodes = torch.min(nodes_arr), torch.max(nodes_arr)
        for n_nodes in range(int(min_nodes), int(max_nodes) + 1):
            idxs = nodes_arr == n_nodes
            values_filtered = values[idxs]
            if len(values_filtered) > 0:
                probs, params = self._create_prob_given_nodes(values_filtered)
                distribution[n_nodes] = {"probs": probs, "params": params}

    def _create_prob_given_nodes(self, values):
        n_bins = self.num_bins  # min(self.num_bins, len(values))
        prop_min, prop_max = torch.min(values), torch.max(values)
        prop_range = prop_max - prop_min + 1e-12
        histogram = torch.zeros(n_bins)
        for val in values:
            i = int((val - prop_min) / prop_range * n_bins)
            # Because of numerical precision, one sample can fall in bin int(n_bins) instead of int(n_bins-1)
            # We move it to bin int(n_bind-1 if that happens)
            if i == n_bins:
                i = n_bins - 1
            histogram[i] += 1
        probs = histogram / torch.sum(histogram)
        try:
            probs = Categorical(probs)
        except:
            probs = Categorical(torch.tensor(probs))
        params = [prop_min, prop_max]
        return probs, params

    def normalize_tensor(self, tensor, prop):
        assert self.normalizer is not None
        mean = self.normalizer[prop]["mean"]
        mad = self.normalizer[prop]["mad"]
        return (tensor - mean) / mad

    def sample(self, n_nodes=19):
        vals = []
        for prop in self.properties:
            dist = self.distributions[prop][n_nodes]
            idx = dist["probs"].sample((1,))
            val = self._idx2value(idx, dist["params"], len(dist["probs"].probs))
            val = self.normalize_tensor(val, prop)
            vals.append(val)
        vals = torch.cat(vals)
        return vals

    def sample_specific(self, prop, n_nodes=19):
        dist = self.distributions[prop][n_nodes]
        idx = dist["probs"].sample((1,))
        val = self._idx2value(idx, dist["params"], len(dist["probs"].probs))
        val = self.normalize_tensor(val, prop)
        return val

    def sample_batch(self, nodesxsample):
        vals = []
        for n_nodes in nodesxsample:
            vals.append(self.sample(int(n_nodes)).unsqueeze(0))
        vals = torch.cat(vals, dim=0)
        return vals

    def _idx2value(self, idx, params, n_bins):
        prop_range = params[1] - params[0]
        left = float(idx) / n_bins * prop_range + params[0]
        right = float(idx + 1) / n_bins * prop_range + params[0]
        val = torch.rand(1) * (right - left) + left
        return val

    def calculate_n_nodes(self, dataset):
        n_nodes = torch.tensor(
            [
                int(j - i)
                for i, j in zip(
                    dataset.slices["x"],
                    dataset.slices["x"][1:],
                )
            ]
        ).long()
        return n_nodes


def prepare_context(properties_list, properties_norm, batch, dataset="aqm"):
    batch_size = len(batch.batch.unique())
    device = batch.x.device
    n_nodes = batch.pos.size(0)
    context_node_nf = 0
    context_list = []
    if dataset == "aqm":
        prop_to_idx = PROP_TO_IDX_AQM
    elif dataset == "aqm_qm7x":
        prop_to_idx = PROP_TO_IDX_AQM_QM7X
    elif dataset == "qm9":
        prop_to_idx = PROP_TO_IDX_QM9
    elif dataset == "geomqm":
        prop_to_idx = PROP_TO_IDX_GEOMQM
    for key in properties_list:
        mean = properties_norm[key]["mean"].to(device)
        std = properties_norm[key]["mad"].to(device)
        properties = batch.y[:, prop_to_idx[key]]
        properties = (properties - mean) / std
        if len(properties) == batch_size:
            # Global feature.
            reshaped = properties[batch.batch]
            if reshaped.size() == (n_nodes,):
                reshaped = reshaped.unsqueeze(1)
            context_list.append(reshaped)
            context_node_nf += 1
        elif len(properties) == n_nodes:
            # Node feature.
            if properties.size() == (n_nodes,):
                properties = properties.unsqueeze(1)
            context_key = properties

            context_list.append(context_key)
            context_node_nf += context_key.size(1)
        else:
            raise ValueError("Invalid tensor size, more than 3 dimensions.")
    # Concatenate
    context = torch.cat(context_list, dim=1)
    assert context.size(1) == context_node_nf
    return context


class ConditionalDistributionNodes:
    def __init__(self, histogram):
        histogram = torch.tensor(histogram).float()
        histogram = histogram + 1e-3  # for numerical stability

        prob = histogram / histogram.sum()

        self.idx_to_n_nodes = torch.tensor(
            [[(i, j) for j in range(prob.shape[1])] for i in range(prob.shape[0])]
        ).view(-1, 2)

        self.n_nodes_to_idx = {
            tuple(x.tolist()): i for i, x in enumerate(self.idx_to_n_nodes)
        }

        self.prob = prob
        self.m = torch.distributions.Categorical(self.prob.view(-1), validate_args=True)

        self.n1_given_n2 = [
            torch.distributions.Categorical(prob[:, j], validate_args=True)
            for j in range(prob.shape[1])
        ]
        self.n2_given_n1 = [
            torch.distributions.Categorical(prob[i, :], validate_args=True)
            for i in range(prob.shape[0])
        ]

        # entropy = -torch.sum(self.prob.view(-1) * torch.log(self.prob.view(-1) + 1e-30))
        entropy = self.m.entropy()
        print("Entropy of n_nodes: H[N]", entropy.item())

    def sample(self, n_samples=1):
        idx = self.m.sample((n_samples,))
        num_nodes_lig, num_nodes_pocket = self.idx_to_n_nodes[idx].T
        return num_nodes_lig, num_nodes_pocket

    def sample_conditional(self, n1=None, n2=None):
        assert (n1 is None) ^ (n2 is None), "Exactly one input argument must be None"

        m = self.n1_given_n2 if n2 is not None else self.n2_given_n1
        c = n2 if n2 is not None else n1

        return torch.tensor([m[i].sample() for i in c], device=c.device)

    def log_prob(self, batch_n_nodes_1, batch_n_nodes_2):
        assert len(batch_n_nodes_1.size()) == 1
        assert len(batch_n_nodes_2.size()) == 1

        idx = torch.tensor(
            [
                self.n_nodes_to_idx[(n1, n2)]
                for n1, n2 in zip(batch_n_nodes_1.tolist(), batch_n_nodes_2.tolist())
            ]
        )

        # log_probs = torch.log(self.prob.view(-1)[idx] + 1e-30)
        log_probs = self.m.log_prob(idx)

        return log_probs.to(batch_n_nodes_1.device)

    def log_prob_n1_given_n2(self, n1, n2):
        assert len(n1.size()) == 1
        assert len(n2.size()) == 1
        log_probs = torch.stack(
            [self.n1_given_n2[c].log_prob(i.cpu()) for i, c in zip(n1, n2)]
        )
        return log_probs.to(n1.device)

    def log_prob_n2_given_n1(self, n2, n1):
        assert len(n2.size()) == 1
        assert len(n1.size()) == 1
        log_probs = torch.stack(
            [self.n2_given_n1[c].log_prob(i.cpu()) for i, c in zip(n2, n1)]
        )
        return log_probs.to(n2.device)
