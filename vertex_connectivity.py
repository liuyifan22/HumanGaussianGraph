import pickle
import numpy as np
import torch

smpl_path = './mvhuman_24/200125/smpl_param/000005.pkl'
with open(smpl_path, 'rb') as f:
    data = pickle.load(f)

faces = data["faces"]


def compute_vertex_neighbors(faces, num_vertices=6890, max_neighbors=30):
    """
    Given a (num_faces, 3) array of face indices (0-indexed) and total number of vertices,
    compute for each vertex (labeled from 1 to num_vertices) the union of neighbors via
    shared connectivity. Two vertices are considered connected if they appear in the same face,
    and two vertices are neighbours if they share at least one connected vertex.
    
    Returns:
        neighbor_tensor (torch.Tensor): Tensor of shape [6890, max_neighbors], where each row
                                          contains the ordered neighbor indices (1-indexed) for that vertex.
                                          If a vertex has less than max_neighbors, the row is padded with 0.
        mask_tensor (torch.Tensor): Boolean Tensor of shape [6890, max_neighbors] where True indicates
                                    a valid neighbor and False indicates padding.
    """
    # Build direct connectivity graph (use 1-indexing for vertices)
    direct_neighbors = {i: set() for i in range(0, num_vertices+1)}
    for face in faces:
        # Convert face vertices from 0-indexed to 1-indexed.
        v = [int(face[i]) + 1 for i in range(3)]
        # For each pair in the triangle, add connections in both directions.
        for i in range(3):
            for j in range(3):
                if i != j:
                    direct_neighbors[v[i]].add(v[j])
    
    # For each vertex, compute the union of neighbors-of-neighbors and remove the vertex itself.
    vertex_neighbors = {}
    for v in range(1, num_vertices+1):
        nbrs = set()
        for u in direct_neighbors[v]:
            nbrs.update(direct_neighbors[u])
        nbrs.discard(v)
        vertex_neighbors[v] = sorted(nbrs)
    
    # Initialize neighbor array and mask.
    neighbor_array = np.zeros((num_vertices, max_neighbors), dtype=np.int32)
    attn_mask = np.ones((num_vertices, max_neighbors), dtype=bool)  # False means valid.
    
    for idx, v in enumerate(range(1, num_vertices+1)):
        nbr_list = vertex_neighbors[v]
        n_valid = min(len(nbr_list), max_neighbors)
        if n_valid > 0:
            neighbor_array[idx, :n_valid] = nbr_list[:n_valid]
            attn_mask[idx, :n_valid] = False
        # If no neighbors or less than max_neighbors, remaining entries stay as 0 (padding) and True.
    
    neighbor_tensor = torch.tensor(neighbor_array, dtype=torch.int32) -1  # Convert to 0-indexing.
    mask_tensor = torch.tensor(attn_mask, dtype=torch.bool)
    return neighbor_tensor, mask_tensor


neighbors, mask = compute_vertex_neighbors(faces)
print("Neighbors tensor shape:", neighbors.shape)  # Expected: [6890, 30]
print("Mask tensor shape:", mask.shape)            # Expected: [6890, 30]

output_path = './vertex_connectivity_tensors.pt'
torch.save({'neighbors': neighbors, 'mask': mask}, output_path)
print(f"Saved tensors to {output_path}")

import pdb; pdb.set_trace()