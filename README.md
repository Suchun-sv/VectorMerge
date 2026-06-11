# VectorMerge 🚀

# Setting Data
We provide embeddings in the HF(
https://github.com/DBgroup-Edinburgh/VectorBenchmark), you can also use https://github.com/DBgroup-Edinburgh/VectorBenchmark)

# Generate Embeddings
## vectormerge generate-embedding

# Create Reference
## vectormerge create-reference
vectormerge create-reference -d [dataset_name] Optional(-m1 [model_name] -m2 [model_name] --m1--m2 [replace default m1 and m2]) --reference-strategy [reference_strategy] --reference-path [reference_path]

```
if reference strategy is random, we do not need to specify m1 and m2, because we can do it randomly.
if reference strategy is ours, we need to specify m1 and m2, because we need to do some clustering
the reference will output to the reference_path, 
```

# Cluster Reference

vectormerge cluster-reference 
-d [dataset_name] 
--reference-key -rk [reference_key]
--reference-path [reference_path] 
--cluster-method [cluster_method] 
--cluster-path [cluster_path]
--force [force retrain, default is use existing cluster]


# Map Embedding [Core]
## vectormerge map-embedding
vectormerge map-embedding -m1 [model_name] -m2 [model_name] --m1--m2 [replace default m1 and m2] -d [dataset_name] --load-reference-path [reference_path] --mapping-method [mapping_method] --mapping-config [mapping_config] --output-path [output_path]

-mapping-method [mapping_method]
-m1 [model_name]
-m2 [model_name]
--m1-m2 [replace default m1 and m2]
-d [dataset_name]
--reference-key [reference_key]
--load-reference-path [reference_path]
--mapping-config [mapping_config]
--output-path [output_path]
--force [force retrain, default is use existing mapping]


# Evaluate
## vectormerge evaluate

vectormerge evaluate 
-d [dataset_name]
-m1 [model_name]
-m2 [model_name]
--reference-key [reference_key]
--mapping-method [mapping_method]
--m1-m2 [replace default m1 and m2]
--reference-path [reference_path]
--mapping-config [mapping_config]
--output-path [output_path]
--force [force retrain, default is use existing mapping]
**kwargs**
---
If you think this paper is useful for your work, please cite us at 
```code
@article{Yang2025integrating,
  author       = {Beining Yang and
                  Yang Cao and
                  Yang Ren},
  title        = {Integrating Vector Databases across Embedding Models},
  journal      = {Proc. {ACM} Manag. Data},
  volume       = {3},
  number       = {6},
  pages        = {1--28},
  year         = {2025}
}
```
