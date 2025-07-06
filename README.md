# VectorMerge 🚀

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

# Map Embedding [Core]
## vectormerge map-embedding
vectormerge map-embedding -m1 [model_name] -m2 [model_name] --m1--m2 [replace default m1 and m2] -d [dataset_name] --load-reference-path [reference_path] --mapping-method [mapping_method] --mapping-config [mapping_config] --output-path [output_path]


# Evaluate
## vectormerge evaluate