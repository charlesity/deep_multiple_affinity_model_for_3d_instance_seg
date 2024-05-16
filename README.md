# Deep Multiple Affinity Model for 3D Instance Segmentation
- An extension of the Proposal-free instance segmentation from Latent Single-Instance Masks
- Based on Dissimilarity Coefficient network (DISCO Net) for learning multiple embedding for pixel affinities and **Active learning** for efficient training
  - See DISCONet class in LSIMasks/models/:
- Instances are preprocessed via graph partitioning once affinities are predicted from voxels.



