# CircWalk: a novel approach to predict CircRNA-disease association based on heterogeneous network representation learning #

## Abstract ##

### Background ###
Several types of RNA in the cell are usually involved in biological processes with multiple functions. Coding RNAs code for proteins while non-coding RNAs regulate gene expression. Some single-strand RNAs can create a circular shape via the back splicing process and convert into a new type called circular RNA (circRNA). circRNAs are among the essential non-coding RNAs in the cell that involve multiple disorders. One of the critical functions of circRNAs is to regulate the expression of other genes through sponging micro RNAs (miRNAs) in diseases. This mechanism, known as the competing endogenous RNA (ceRNA) hypothesis, and additional information obtained from biological datasets can be used by computational approaches to predict novel associations between disease and circRNAs.

### Results ###
We applied multiple classifiers to validate the extracted features from the heterogeneous network and selected the most appropriate one based on some evaluation criteria. Then, the XGBoost is utilized in our pipeline to generate a novel approach, called CircWalk, to predict CircRNA-Disease associations. Our results demonstrate that CircWalk has reasonable accuracy and AUC compared with other state-of-the-art algorithms. We also use CircWalk to predict novel circRNAs associated with lung, gastric, and colorectal cancers as a case study. The results show that our approach can accurately detect novel circRNAs related to these diseases.

### Conclusions ###
Considering the ceRNA hypothesis, we integrate multiple resources to construct a heterogeneous network from circRNAs, mRNAs, miRNAs, and diseases. Next, the DeepWalk algorithm is applied to the network to extract feature vectors for circRNAs and diseases. The extracted features are used to learn a classifier and generate a model to predict novel CircRNA-Disease associations. Our approach uses the concept of the ceRNA hypothesis and the miRNA sponge effect of circRNAs to predict their associations with diseases. Our results show that this outlook could help identify CircRNA-Disease associations more accurately.

The twelve datasets we used can be found at the following web addresses, respectively: Circ2Disease at http://bioinformatics.zju.edu.cn/Circ2Disease , CircR2Disease at http://bioinfo.snnu.edu.cn/CircR2Disease , CTD at https://ctdbase.org , circAtlas at http://circatlas.biols.ac.cn/ , circBase at http://www.circbase.org/ ,  RAID at http://www.rna-society.org/404.shtml , starBase at http://www.sysu.edu.cn/403.html , HMDD at http://www.cuilab.cn/hmdd , miR2Disease at http://www.mir2disease.org/ , miRTarBase at https://mirtarbase.cuhk.edu.cn/~miRTarBase/miRTarBase_2022/php/index.php , DisGeNET at https://www.disgenet.org/ , and MeSH at https://www.nlm.nih.gov/mesh/meshhome.html

To cite the article, you can use the citation below:
Kouhsar, M., Kashaninia, E., Mardani, B. et al. CircWalk: a novel approach to predict CircRNA-disease association based on heterogeneous network representation learning. BMC Bioinformatics 23, 331 (2022). https://doi.org/10.1186/s12859-022-04883-9


### Notes ###
Before you run the code, either:
  - download the files listed [here](https://www.dropbox.com/scl/fo/tcodexrgvnx81ext0x8uf/h?dl=0&rlkey=f0l9hlzyhg2cy8sfwzgnkde2s) and paste them into the `Data` directory of our project.
Or,
  - install [deepwalk]() and do the following:
    1. Go to the RawData directory on the clone of the CircWalk repo that you have downloaded on your machine. Then, run this command for every feature space size you need:

``` deepwalk --format edgelist --input intMergedNetwork.edgelist --representation-size <FEATURE_SIZE> --workers 2 --output entity_representations<FEATURE_SIZE>.embeddings ```

E.g. to make embeddings (=feature vectors) of size 100 this is what the command would look like:

``` deepwalk --format edgelist --input intMergedNetwork.edgelist --representation-size 100 --workers 2 --output entity_representations100.embeddings
(That --workers parameter is optional but it obviously increases performance) ```

  2. Then remove the first row of the output file named entity_representations<FEATURE_SIZE>.embeddings. The first row consists of only two integers and is not a feature vector and hence has to be removed becuase in the Python code it is assumed that embedding files are .csv files consisting of fixed-length rows of features.
  As you can observe in the code, the current code needs feature sizes from 10 to 200 (only the multiples of ten)

  3. Then you can cut and paste the output files to the Data directory where embeddings of size 10 to 50 already reside and run boosting_training.py for the XGBoost classifier, or any other file in /Data ending with "_training.py" for other classifiers. The results will be written to /Results/xgboost/xgboostReport.docx.
