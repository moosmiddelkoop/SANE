|     |     | Towards |                        | Scalable |     | and | Versatile           | Weight | Space        | Learning |     |     |     |     |
| --- | --- | ------- | ---------------------- | -------- | --- | --- | ------------------- | ------ | ------------ | -------- | --- | --- | --- | --- |
|     |     |         | KonstantinSchu¨rholt12 |          |     |     | MichaelW.Mahoney234 |        | DamianBorth1 |          |     |     |     |     |
Abstract
| Learning |     | representations |     | of well-trained |     | neural |     |     |     |     |     |     |     |     |
| -------- | --- | --------------- | --- | --------------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
4202 nuJ 41  ]GL.sc[  1v79990.6042:viXra networkmodelsholdsthepromisetoprovidean
understandingoftheinnerworkingsofthosemod-
els. However,previousworkhaseitherfacedlim-
itationswhenprocessinglargernetworksorwas
task-specifictoeitherdiscriminativeorgenerative
ThispaperintroducestheSANEapproach
tasks.
| toweight-spacelearning. |             |     |     | SANEovercomespre- |               |      |     |     |     |     |     |     |     |     |
| ----------------------- | ----------- | --- | --- | ----------------- | ------------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
| vious                   | limitations |     | by  | learning          | task-agnostic | rep- |     |     |     |     |     |     |     |     |
resentationsofneuralnetworksthatarescalable
| to  | larger | models | of  | varying | architectures | and |     |                    |     |            |                |     |         |         |
| --- | ------ | ------ | --- | ------- | ------------- | --- | --- | ------------------ | --- | ---------- | -------------- | --- | ------- | ------- |
|     |        |        |     |         |               |     |     | Figure1.Aggregated |     | results of | 56 experiments |     | showing | (left:) |
thatshowcapabilitiesbeyondasingletask. Our fourdiscriminativedownstreamtasksinR2,and(right:)fourgen-
methodextendstheideaofhyper-representations
erativedownstreamtasksinaccuracy,eachevaluatedon(bottom:)
towardssequentialprocessingofsubsetsofneu-
CNNsmodelzoostrainedon4datasets(NMIST,SVHN,CIFAR-
ral network weights, thus allowing one to em- 10, STL) and (top:) ResNet18 model zoos trained on three
bedlargerneuralnetworksasasetoftokensinto datasets(CIFAR-10,CIFAR-100,Tiny-ImageNet).Thecolorsin-
|     |         |                |     |        | SANEreveals |     |     | dicateperformanceofRed:rawNNweights,Orange:weight |     |     |     |     |     |     |
| --- | ------- | -------------- | --- | ------ | ----------- | --- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| the | learned | representation |     | space. |             |     |     |                                                   |     |     |     |     |     |     |
statisticsfromUnterthineretal.(2020),Green:
| global | model |     | information | from | layer-wise | em- |     |     |     |     |     |     | trainedhyper- |     |
| ------ | ----- | --- | ----------- | ---- | ---------- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- |
beddings, and it can sequentially generate un- representationsfromSchu¨rholtetal.(2021;2022a),andBlue:
SANE(ours).Whilesomemethodsperformwellonspecifictasks,
seenneuralnetworkmodels,whichwasunattain-
orarerestrictedbythesizeoftheunderlyingmodels,SANEcan
ablewithprevioushyper-representationlearning
deliverexcellentperformanceonalltasksandmodelsizes.
| methods. |     | Extensiveempiricalevaluationdemon- |     |     |     |     |     |                       |     |          |          |       |     |         |
| -------- | --- | ---------------------------------- | --- | --- | --- | --- | --- | --------------------- | --- | -------- | -------- | ----- | --- | ------- |
|          |     |                                    |     |     |     |     |     | In the discriminative |     | context, | previous | works | aim | to link |
stratesthatSANEmatchesorexceedsstate-of-the-
artperformanceonseveralweightrepresentation weightspacepropertiestopropertiessuchasmodelqual-
ity,generalizationgap,orhyperparameters,usingeitherthe
learningbenchmarks,particularlyininitialization
fornewtasksandlargerResNetarchitectures. margin distribution (Yak et al., 2019; Jiang et al., 2019),
graphtopologyfeatures(Corneanuetal.,2020),oreigen-
|     |     |     |     |     |     |     |     | value decompositions |     | of weight | matrices |     | (Martin | & Ma- |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------- | --- | --------- | -------- | --- | ------- | ----- |
honey,2019b;2020;2021;Martinetal.,2021).Someworks
1.Introduction
|     |     |     |     |     |     |     |     | learn classifiers | to  | map between | statistics |     | of weights | and |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | ----------- | ---------- | --- | ---------- | --- |
modelproperties(Eilertsenetal.,2020;Unterthineretal.,
| The exploration |     | of  | the “weight | space” | of  | neural | network |                                                   |     |     |     |     |     |     |
| --------------- | --- | --- | ----------- | ------ | --- | ------ | ------- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|                 |     |     |             |        |     |        |         | 2020), orlearnlower-dimensionalmanifoldstoinferNN |     |     |     |     |     |     |
(NN)models,i.e.,thehigh-dimensionalspacespannedby
modelproperties(Schu¨rholtetal.,2021).
themodelparametersofapopulationoftrainedNNs,allows
ustogaininsightsintotheinnerworkingsofthosemodels. In the generative context, methods have been proposed
togeneratemodelweightsusing(Graph)HyperNetworks
1AIML
|     | Lab, | University |     | of St.Gallen, | St. | Gallen, | Switzer- |     |     |     |     |     |     |     |
| --- | ---- | ---------- | --- | ------------- | --- | ------- | -------- | --- | --- | --- | --- | --- | --- | --- |
2International (Haetal.,2016;Zhangetal.,2019;Knyazevetal.,2021),
| land      |     | Computer |          | Science     | Institute, | Berkeley, | CA, |                        |     |           |     |        |           |     |
| --------- | --- | -------- | -------- | ----------- | ---------- | --------- | --- | ---------------------- | --- | --------- | --- | ------ | --------- | --- |
| 3Lawrence |     |          |          |             |            |           |     | Bayesian HyperNetworks |     | (Deutsch, |     | 2018), | HyperGANs |     |
| USA       |     | Berkeley | National | Laboratory, |            | Berkeley, | CA, |                        |     |           |     |        |           |     |
USA4DepartmentofStatistics,UniversityofCaliforniaatBerke- (Ratzlaff&Fuxin,2019),andHyperTransformers(Zhmogi-
ley,CA,USA.Correspondenceto: KonstantinSchu¨rholt<kon- novetal.,2022).Theseapproacheshavebeenusedfortasks
stantin.schuerholt@unisg.ch>. suchasneuralarchitecturesearch,modelcompression,en-
|             |     | 41st |               |     |            |     |         | sembling,transferlearning,andmeta-learning. |     |     |     |     | Theyhave |     |
| ----------- | --- | ---- | ------------- | --- | ---------- | --- | ------- | ------------------------------------------- | --- | --- | --- | --- | -------- | --- |
| Proceedings | of  | the  | International |     | Conference | on  | Machine |                                             |     |     |     |     |          |     |
incommonthattheyderivetheirlearningsignalfromthe
Learning,Vienna,Austria.PMLR235,2024.Copyright2024by
|     |     |     |     |     |     |     |     | underlying(typicallyimage) |     |     | dataset. | Incontrasttothese |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------------- | --- | --- | -------- | ----------------- | --- | --- |
theauthor(s).
1

TowardsScalableandVersatileWeightSpaceLearning
Givenmodelzoostrainedondifferentclassificationtasks,weextractandsequentializethemodelweights.SANEtrainshyper-
Figure2.
representationsonweightssubsequences,i.e.,individuallayers.SANEcanbeusedformultipledownstreamtasks,eitherusingtheencoder
fordiscriminativetaskssuchasthepredictionofmodelaccuracy,orthedecoderforgenerativetaskssuchassamplingofnewmodels.
methods,so-calledhyper-representations(Schu¨rholtetal., forheld-outNNmodelsofthemodelzoousedfortraining
2022a) learn a lower-dimensional representation directly SANEbutalsoforNNmodelsofout-of-distributionmodel
fromtheweightspacewithouttheneedtohaveaccessto zooswithdifferentarchitecturesandtrainingdata. Further,
wedemonstratethatSANEcanlearnhyper-representations
data,e.g.,theimagedataset,tosampleunseenNNmodels
fromthatlatentrepresentation. of much larger NN models, and so it makes them appli-
|                                 |     |                   |     |     |             |          |        | cabletoreal-worldproblems. |                | Inparticular,themodelsin |          |           |            |
| ------------------------------- | --- | ----------------- | --- | --- | ----------- | -------- | ------ | -------------------------- | -------------- | ------------------------ | -------- | --------- | ---------- |
| Inthispaper,wepresentSequential |     |                   |     |     | Autoencoder |          |        |                            |                |                          |          |           |            |
|                                 |     |                   |     |     |             |          |        | the ResNet                 | model zoo used | for                      | training | are three | orders     |
| for Neural                      |     | Embeddings(SANE), |     |     | an          | approach | to     |                            |                |                          |          |           |            |
|                                 |     |                   |     |     |             |          |        | of magnitudes              | larger than    | all model                | zoos     | used      | for hyper- |
| learn task-agnostic             |     | representations   |     |     | of NN       | weight   | spaces |                            |                |                          |          |           |            |
representation
|     |     |     |     |     |     |     |     |     | learning in | previous | works. | While | previ- |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----------- | -------- | ------ | ----- | ------ |
capableofembeddingindividualNNmodelsintoalatent
oushyper-representationlearningmethodswerestructurally
| space to | perform | the | above-mentioned |     | discriminative |     | or  |     |     |     |     |     |     |
| -------- | ------- | --- | --------------- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
constrainedtoencodetheentireNNmodelatonce,SANEis
| generative | downstream |     | tasks. | Our | approach | builds | upon |     |     |     |     |     |     |
| ---------- | ---------- | --- | ------ | --- | -------- | ------ | ---- | --- | --- | --- | --- | --- | --- |
scalablebyapplyingitssequentialapproachtoencodelay-
| the idea | of hyper-representations |     |     | (Schu¨rholt |     | et al., | 2021; |     |     |     |     |     |     |
| -------- | ------------------------ | --- | --- | ----------- | --- | ------- | ----- | --- | --- | --- | --- | --- | --- |
ersorsubsetsofweightsintohyper-representationembed-
| 2022a),                                     | which | learn | a lower-dimensional |     |                    | representation |         |                                                    |                |         |     |               |     |
| ------------------------------------------- | ----- | ----- | ------------------- | --- | ------------------ | -------------- | ------- | -------------------------------------------------- | -------------- | ------- | --- | ------------- | --- |
|                                             |       |       |                     |     |                    |                |         | dings. While                                       | we demonstrate | scaling | up  | to ResNet-101 |     |
| z fromapopulationofNNmodels.                |       |       |                     |     | Thisisaccomplished |                |         |                                                    |                |         |     |               |     |
|                                             |       |       |                     |     |                    |                |         | models,SANEisnotfundamentallylimitedtothatsize.    |                |         |     |               | Fi- |
| byauto-encodingtheirflattenedweightvectorsw |       |       |                     |     |                    |                | through |                                                    |                |         |     |               |     |
|                                             |       |       |                     |     |                    | i              |         | nally,weevaluateSANEonbothdiscriminativeandgenera- |                |         |     |               |     |
atransformerarchitecture,withthebottleneckactingasa
|                             |     |     |            |                |       |       |        | tivedownstreamtasks.                                | Fordiscriminatetasks,weevaluate |     |     |     |     |
| --------------------------- | --- | --- | ---------- | -------------- | ----- | ----- | ------ | --------------------------------------------------- | ------------------------------- | --- | --- | --- | --- |
| lower-dimensionalembeddingz |     |     |            | ofeachNNmodel. |       |       | While  |                                                     |                                 |     |     |     |     |
|                             |     |     |            | i              |       |       |        | onsixmodelzoosbylinear-probingforpropertiesoftheun- |                                 |     |     |     |     |
| the hyper-representation    |     |     | method     | promises       |       | to be | useful |                                                     |                                 |     |     |     |     |
|                             |     |     |            |                |       |       |        | derlyingNNmodels.                                   | Forgenerativetasks,weevaluateon |     |     |     |     |
| for discriminative          |     | and | generative | tasks,         | until | now,  | sepa-  |                                                     |                                 |     |     |     |     |
sevenmodelzoosbysamplingtargetedmodelweightsfor
| rate hyper-representations |     |     |     | had to be | trained | specifically |     |     |     |     |     |     |     |
| -------------------------- | --- | --- | --- | --------- | ------- | ------------ | --- | --- | --- | --- | --- | --- | --- |
initializationandtransferlearning.
| foreitherdiscriminativeorgenerativetasks. |     |     |     |     |     | Additionally, |     |     |     |     |     |     |     |
| ----------------------------------------- | --- | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- |
existing approaches have a major shortcoming: the un- WeprovideanaggregatedoverviewofourresultsinFig.1.
derlying encoder-decoder model has to embed the entire OnverysmallCNNmodels(evaluatedonMNIST,SVHN,
flattenedweightvectorsw i atonceintothelearnedlower- CIFAR-10, and STL, which we include for comparison
dimensionalrepresentationz.Thisdrasticallylimitsthesize withpriorwork),SANEperformsaswellaspreviousstate-
ofNNsthatcanbeembedded. SANEaddressestheselimi- of-the-art (SOTA) in discriminative tasks. In generative
tationsbydecomposingtheentireweightvectorw intolay- downstream tasks, SANEoutperforms SOTA by 25% in
i
ersorsmallersubsets,andthensequentiallyprocessesthem. accuracyforinitializationonthesametaskand17%inac-
InsteadofencodingtheentireNNmodelbyoneembedding, curacyforfinetuningtonewtasks. Onlargermodelssuch
SANEencodesapotentiallyverylargeNNasmultipleem- as ResNets (evaluated on CIFAR-10, CIFAR-100, Tiny-
beddings. Thechangefromprocessingtheentireflattened ImageNet, which were beyond the capabilities of prior
weightvectortosubsetsofweightsismotivatedbyMartin work),weshowresultscomparabletobaselinesfordiscrim-
&Mahoney(2019a;2021),whoshowedthatglobalmodel inative downstream tasks, and we report outperformance
informationispreservedinthelayer-wisecomponentsof to baselines for generative downstream tasks by 31% for
NNs. AnillustrationofourapproachcanbefoundinFig.2. initialization and 28% for finetuning to new tasks. Addi-
tionally,weshowthatSANEcansampletargetedmodelsby
| To evaluate | SANE, | we  | analyze | how | NN embeddings |     | en- |     |     |     |     |     |     |
| ----------- | ----- | --- | ------- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- |
promptingwithdifferentarchitecturesthanitusedfortrain-
codedbySANEbehaveincomparisontoMartin&Mahoney
|     |     |     |     |     |     |     |     | ing. Thesesampledmodelscanoutperformmodelstrained |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- |
(2019a;2021)qualitymeasures.Weshowthatsomeofthese
|     |     |     |     |     |     |     |     | fromscratchonthepromptedarchitecture. |     |     |     | Codeisavail- |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------- | --- | --- | --- | ------------ | --- |
weightmatrixqualitymetricsshowsimilarcharacteristics
ableatgithub.com/HSG-AIML/SANE.
astheembeddingsproducedbySANE.Thisholdsnotonly
2

TowardsScalableandVersatileWeightSpaceLearning
2.Methods the position of a token, we use a 3-dimensional position
P =[n,l,k],wheren∈[1,N]indicatestheglobalposi-
n
Hyper-representationslearnanencoder-decodermodelon
tioninthesequence,l∈[1,L]indicatethelayerindex,and
theweightsofNNs(Schu¨rholtetal.,2021):
k ∈[1,K(l)]isthepositionofthetokenwithinthelayer.
|     |     |     | z=g θ | (W) |     | (1) |        |          |       |          |       |           |     |
| --- | --- | --- | ----- | --- | --- | --- | ------ | -------- | ----- | -------- | ----- | --------- | --- |
|     |     |     |       |     |     |     | Out of | the full | token | sequence | T and | positions | P ∈ |
W(cid:99) =h (z), (2) NN×3,wetakearandomconsecutivesub-sequenceT =
|     |     |     |     | ψ   |     |     |            |      |           |     |                |     | s,n       |
| --- | --- | --- | --- | --- | --- | --- | ---------- | ---- | --------- | --- | -------------- | --- | --------- |
|     |     |     |     |     |     |     | T          | with | positions | P   | = P            |     | of length |
|     |     |     |     |     |     |     | n,...,n+ws |      |           |     | s,n n,...,n+ws |     |           |
whereg istheencoderwhichmapstheflattenedweights
|                     | θ                                     |     |                            |     |     |     | ws. Wecallthesesub-sequenceswindowsandthelengthof |     |     |     |     |     |     |
| ------------------- | ------------------------------------- | --- | -------------------------- | --- | --- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| Wtoembeddingsz,andh |                                       |     | decodesbacktoreconstructed |     |     |     |                                                   |     |     |     |     |     |     |
|                     |                                       |     | ψ                          |     |     |     | thesub-sequencethewindowsizews.                   |     |     |     |     |     |     |
| weightsW(cid:99).   | Eventhoughpreviousworkrealizedbothen- |     |                            |     |     |     |                                                   |     |     |     |     |     |     |
coderanddecoderwithtransformerbackbones,theweight ForSANEonwindowsoftokens,weextendEqs. 1and2to
encodeanddecodetokenwindowsas
vectorhadtobeoffixedsize,andmodelsarerepresentedin
aglobalembeddingspace(Schu¨rholtetal.,2021;2022a).
|     |     |     |     |     |     |     |     |     | z s,n | =g θ (T | s,n ,P s,n ) |     | (3) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | ------- | ------------ | --- | --- |
Hyper-representationsaretrainedwithareconstructionloss
L = ∥W−W(cid:99)∥2 and contrastive guidance loss L = T(cid:98)s,n =h (z ,P ), (4)
| rec      |      | 2          |             |     |                 | c     |     |     |     | ψ   | s,n s,n |     |     |
| -------- | ---- | ---------- | ----------- | --- | --------------- | ----- | --- | --- | --- | --- | ------- | --- | --- |
| NTXent(p | ϕ (z | i ),p ϕ (z | j )), where | p ϕ | is a projection | head. |     |     |     |     |         |     |     |
∈Rws×dz
Schu¨rholtetal.(2021)proposedweightpermutation,noise, wherez s,n istheper-tokenlatentrepresentation
andmaskingasaugmentationstogenerateviewsi,j ofthe of the window. In contrast to hyper-representations Eqs.
samemodel. 1 and 2 which operate on the full flattened weights of a
model,SANEencodessub-sequencesoftokenizedmodels.
Existinghyper-representationmethodshavetwomajorlim-
|     |     |     |     |     |     |     | For simplicity, |     | we apply | linear | mapping | to and | from the |
| --- | --- | --- | --- | --- | --- | --- | --------------- | --- | -------- | ------ | ------- | ------ | -------- |
itations: i)usingthefullweightvectortocomputeglobal
|     |     |     |     |     |     |     | bottleneck,toreducetokensfromd |     |     |     | t tod | z . |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------ | --- | --- | --- | ----- | --- | --- |
modelembeddingsbecomesinfeasibleforlargermodels;
|              |     |            |        |     |            |            | We  | adapt | the composite |     | training | loss of | hyper- |
| ------------ | --- | ---------- | ------ | --- | ---------- | ---------- | --- | ----- | ------------- | --- | -------- | ------- | ------ |
| and ii) they | can | only embed | models |     | that share | the archi- |     |       |               |     |          |         |        |
tecturewiththeoriginalmodelzoo. OurSANEmethodad- representations,L=(1−γ)L rec +γL c ,forsequencesas:
| dresses | both of | these limitations. |     | To  | make models | more |     |     |     |          |     |          |     |
| ------- | ------- | ------------------ | --- | --- | ----------- | ---- | --- | --- | --- | -------- | --- | -------- | --- |
|         |         |                    |     |     |             |      |     |     |     | (cid:16) |     | (cid:17) |     |
∥2
digestibleforpretrainingandinference,weproposetoex- L rec =∥M s,n ⊙ T s,n −T(cid:98)s,n (5)
2
| pressmodelsassequencesoftokenvectors. |     |     |     |     | Toaddressi), |     |     |     |           |     |        |        |     |
| ------------------------------------- | --- | --- | --- | --- | ------------ | --- | --- | --- | --------- | --- | ------ | ------ | --- |
|                                       |     |     |     |     |              |     |     | L   | =NTXent(p |     | (z ),p | (z )). |     |
SANElearns per-token embeddings, which are trained on c ϕ s,n,i ϕ s,n,j (6)
| subsequencesofthefullbasemodelsequence.        |     |     |     |     |     | Thisway, |               |     |     |                                |     |     |     |
| ---------------------------------------------- | --- | --- | --- | --- | --- | -------- | ------------- | --- | --- | ------------------------------ | --- | --- | --- |
|                                                |     |     |     |     |     |          | Here,themaskM |     |     | indicatessignalwith1andpadding |     |     |     |
| thememoryandcomputeloadaredecoupledfromthebase |     |     |     |     |     |          |               |     | s,n |                                |     |     |     |
with0,toensurethatthelossisonlycomputedonactual
| modelsize. | Bydecouplingthetokenizationfromtherepre- |     |     |     |     |     |     |     |     |     |     |     |     |
| ---------- | ---------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
weights. Thecontrastiveguidancelossusestheaugmented
| sentationlearning,wealsoaddressii).                 |     |     |     |     | Themodelsinthe |     |          |                    |     |     |     |     |     |
| --------------------------------------------------- | --- | --- | --- | --- | -------------- | --- | -------- | ------------------ | --- | --- | --- | --- | --- |
|                                                     |     |     |     |     |                |     | viewsi,j | andprojectionheadp |     |     | .   |     |     |
| modelzoosetcanhavevaryingarchitectures,aslongasthey |     |     |     |     |                |     |          |                    |     |     | ϕ   |     |     |
areexpressedasasequencewiththesametoken-vectorsize. ThepretrainingprocedureisdetailedinAlgorithm1. We
Thetransformerbackboneandper-tokenembeddingsalso preprocess model weights by standardizing weights per
allowchangestothelengthofthesequenceduringorafter layer and aligning all models to a reference model; see
training. Below,weprovidetechnicaldetailsonSANE.We ModelAlignment below. Asinpreviouswork(Schu¨rholt
firstprovidedetailsonpretrainingSANE,computingmodel etal.,2021;Peeblesetal.,2022),theencoderanddecoder
embeddings,andsamplingmodels;andwethenintroduce arerealizedastransformerblocks. Trainingonthefullse-
additionalaligning,haloing,andbn-conditioningmethods quencewouldmemory-limitthebase-modelsizebyitsse-
|     |     |     |     |     |     |     | quencelength. |     | Trainingtheencoderanddecoderonwin- |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ---------------------------------- | --- | --- | --- | --- |
tostabilizetrainingandinference.
| SANE: | Sequential |     | Autoencoder |     | for | Neural |     |     |     |     |     |     |     |
| ----- | ---------- | --- | ----------- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
Algorithm1SANEpretraining
Embeddings
|                 |     | To tokenize                     |     | weights, | we reshape  | the |                             |                    |     |     |     |     |     |
| --------------- | --- | ------------------------------- | --- | -------- | ----------- | --- | --------------------------- | ------------------ | --- | --- | --- | --- | --- |
|                 |     |                                 |     |          |             |     | Input:                      | populationofmodels |     |     |     |     |     |
| weights         | W   | ∈ Rcout×c1×···×cin              |     | to       | 2d matrices | W ∈ |                             |                    |     |     |     |     |     |
|                 | raw |                                 |     |          |             |     | i: standardizemodelsweights |                    |     |     |     |     |     |
| Rcout×cr,wherec |     | aretheoutgoingchannels,andwhere |     |          |             |     |                             |                    |     |     |     |     |     |
out
ii: alignmodelstoonecommonreferencemodel
| c r the remaining, |     | flattened | dimensions. |     | We  | then slice |     |     |     |     |     |     |     |
| ------------------ | --- | --------- | ----------- | --- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- |
iii: tokenizemodelstotokensT,positionsP,masksM
| theweightsrow-wise,alongtheoutgoingchannel. |     |     |     |     |     | Using |     |                       |     |     |      |     |     |
| ------------------------------------------- | --- | --- | --- | --- | --- | ----- | --- | --------------------- | --- | --- | ---- | --- | --- |
|                                             |     |     |     |     |     |       |     | drawkwindowspermodel: |     |     | T ,P | ,M  |     |
globaltokensized ,wesplittheslicesintomultipleparts iv: s,n s,n s,n
t
|     |     |     |     |     |     |     | v: trainonL |     | untilconvergenceofL |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------- | --- | ------------------- | --- | --- | --- | --- |
if c > d and zero-pad to fill up to d . For weights W train val
| r         | t                  |     |     |           | t      | l   |     |     |     |     |     |     |     |
| --------- | ------------------ | --- | --- | --------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
| oflayerl, | thisgivesustokensT |     |     | ∈ Rnl×dt, | wheren | =   |     |     |     |     |     |     |     |
|           |                    |     |     | l         |        | l   |     |     |     |     |     |     |     |
ceil(c
c out,l r ). Since all tokens T l share the same token dowsinsteadofthefullmodelsequencedecouplesthemem-
d t
size, thetokensoflayerl = 1,...,Lcanbeconcatenated oryrequirementfromthebasemodel’sfullsequencelength.
togetthemodeltokensequenceT ∈ RN×dt. Toindicate ThewindowsizecanbeusedtobalanceGPUmemoryload
3

TowardsScalableandVersatileWeightSpaceLearning
andtheamountofcontextinformation. Notably,sincewe putethetokensequenceTeandcorrespondingembedding
disentanglethetokenizationfromtherepresentationlearn- sequence ze = g (Te,P). Following previous work, we
θ
ingmodel,SANEalsoallowsustoembedsequencesofmod-
|     |     |     |     |     | modelthedistributionP |     | withaKernelDensityEstimation |     |     |
| --- | --- | --- | --- | --- | --------------------- | --- | ---------------------------- | --- | --- |
elswithvaryingarchitectures,aslongastheirtokensizeis (KDE) per token as P (ze) (Schu¨rholt et al., 2022a).
|     |     |     |     |     |     |     | e∈E n |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- | --- |
thesame. Topreventpotentialoverfittingtospecificwin- Wethendrawknewtokensamplesas:
dowpositions,weproposetosamplewindowsfromeach
|                                     |     |     |     |     |     | zk  | ∼P    | (ze). | (7) |
| ----------------------------------- | --- | --- | --- | --- | --- | --- | ----- | ----- | --- |
| modelsequencemultipletimesrandomly. |     |     |     |     |     |     | n e∈E | n     |     |
Wereconstructthesampledembeddingstoweighttokens
| ComputingSANEModelEmbeddings.                      |     |     |     | SANEcanbeused |                                |     |     |                |     |
| -------------------------------------------------- | --- | --- | --- | ------------- | ------------------------------ | --- | --- | -------------- | --- |
|                                                    |     |     |     |               | Tk = h (zk,P)andthenweightsWk. |     |     | Samplingtokens |     |
| toanalyzemodelsinembeddingspace,e.g.,byusingembed- |     |     |     |               | ψ                              |     |     |                |     |
canbedonecheaply,decodingandevaluatingtheweightsus-
dingsasfeaturestopredictpropertiessuchasaccuracyor
ingsomeperformancemetricinvolvesonlyforwardpasses
| toidentifyothermodelqualitymetrics. |     |     |     | Incontrasttohyper- |                 |        |            |              |         |
| ----------------------------------- | --- | --- | --- | ------------------ | --------------- | ------ | ---------- | ------------ | ------- |
|                                     |     |     |     |                    | and is likewise | cheap. | Therefore, | one can draw | a large |
representations,SANEcanembeddifferentmodelsizesand
amountofsamplesandkeeponlythetopmmodels,accord-
| architecturesinthesameembeddingspace. |     |     |     | Toembedany |                            |     |     |                         |     |
| ------------------------------------- | --- | --- | --- | ---------- | -------------------------- | --- | --- | ----------------------- | --- |
|                                       |     |     |     |            | ingtotheperformancemetric. |     |     | Wecallthismethodsubsam- |     |
model,webeginbypreprocessingweightsbystandardiz-
pling. Theprocesscanberefinediteratively,byre-usingthe
| ingperlayerandaligningmodelstoapre-definedreference |     |     |     |     | embeddingszk |     |     |     |     |
| --------------------------------------------------- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- |
ofthebestmodelsasnewpromptexamples,
| model(seeModelAlignmentbelow). |     |     |     | Subsequently,thepre- |     |     |     |     |     |
| ------------------------------ | --- | --- | --- | -------------------- | --- | --- | --- | --- | --- |
toadjustthesamplingdistributiontobestfittheneedsof
| processed | models | are tokenized | as described | above. For |                       |     |                               |     |     |
| --------- | ------ | ------------- | ------------ | ---------- | --------------------- | --- | ----------------------------- | --- | --- |
|           |        |               |              |            | theperformancemetric. |     | Wecallthissamplingmethodboot- |     |     |
shortmodelsequences,theembeddingsequencescanbedi-
|                                       |     |     |                              |             | strapped. ByonlyrequiringaroughversionofP            |     |     |     | andrefin- |
| ------------------------------------- | --- | --- | ---------------------------- | ----------- | ---------------------------------------------------- | --- | --- | --- | --------- |
| rectlycomputedasz                     |     | =   | g (T,P). Forlargermodels,the |             |                                                      |     |     |     |           |
|                                       |     |     | θ                            |             | ingwiththetargetsignal,oursamplingstrategyreducesre- |     |     |     |           |
| tokensequencesaretoolongtoembedasone. |     |     |                              | Wetherefore |                                                      |     |     |     |           |
quirementsonpromptexamplessuchthatonlyveryfewand
employhaloing(seebelow)toencodetheentiresequence
|                         |     |                               |                            |     | slightlytrainedpromptexamplesarenecessary. |     |     |     | Theover- |
| ----------------------- | --- | ----------------------------- | -------------------------- | --- | ------------------------------------------ | --- | --- | --- | -------- |
| ascoherentsubsequences. |     |                               | Algorithm2summarizestheem- |     |                                            |     |     |     |          |
|                         |     |                               |                            |     | allsamplingmethodisoutlinedinAlgorithm3.   |     |     |     | Itmakes  |
| beddingcomputation.     |     | Tocomparedifferentmodelsinem- |                            |     |                                            |     |     |     |          |
useofmodelalignment,haloing,andbatch-normcondition-
Algorithm2SANEmodelembeddingcomputation ingwhicharedetailedbelow. Inadditiontothecomputeef-
Input: populationofmodels ficiency,thesesamplingmethodslearnthedistributionof
i: preprocessing: standardizeandalignmodelweights targetedmodelsinembeddingspace. Further,theyarenot
boundtothedistributionofpromptexamples,butinstead
| ii: | tokenizemodels: | T,positionsP,propertyy |     |     |     |     |     |     |     |
| --- | --------------- | ---------------------- | --- | --- | --- | --- | --- | --- | --- |
iii: splitT,PtoconsecutivechunksT ,P theycanfindthedistributionthatbestsatisfiesthetargetper-
hs,n hs,n
formancemetric,independentofthepromptexamples.
| iv: | computeembeddingsz |     | hs,n =g | θ (T hs,n ,P hs,n ) |     |     |     |     |     |
| --- | ------------------ | --- | ------- | ------------------- | --- | --- | --- | --- | --- |
v:stitchmodelembeddingsztogetherfromchunksz
hs,n Algorithm3SamplingmodelswithSANE
beddingspace,weaggregatethesequencesoftokenembed- Input: modelpromptexamplesWe
dings. Tothatend,weunderstandthetokensequenceofone tokensTe,positionsPe
i: tokenizepromptexamples:
modeltoformasurfaceinembeddingspaceandchooseto ii: embedpromptexampleszefollowingAlg. 2
representthatsurfacebyitscenterofgravity. Thatis,we fori =1tobootstrapiterationsdo
boot
takethemeanofalltokensalongtheembeddingdimension iii: drawksampleszk ∼P (ze)
|     | (cid:80)N |     |     |     |     |     | n   | e∈ E n |     |
| --- | --------- | --- | --- | --- | --- | --- | --- | ------ | --- |
as¯z= 1 (z ). Thatresultsinonevectorinembed- iv: decodetotokensT k =h (zk )
|                    | N n=1 | n                               |     |     |     |     |     | ψ   |     |
| ------------------ | ----- | ------------------------------- | --- | --- | --- | --- | --- | --- | --- |
| dingspacepermodel. |       | Ofcourse,onecoulduseotheraggre- |     |     |     |     |     |     |     |
v: applybatch-normconditioning
gationmethodswithSANE. vi: computetargetmetricandkeepbestmmodels
ifbootstrapiterations>1then
| SamplingModelswithFewPromptExamples. |     |     |     | Sampling |      |           |      |     |     |
| ------------------------------------ | --- | --- | --- | -------- | ---- | --------- | ---- | --- | --- |
|                                      |     |     |     |          | vii: | ze =zkfor | k ∈m |     |     |
modelsfromSANEpromisestotransferknowledgefromex-
endif
istingpopulationstonewmodelswithdifferentarchitectures.
| Givenpretrainedencodersg     |     |     | anddecoderh            |                 | endfor |     |     |     |     |
| ---------------------------- | --- | --- | ---------------------- | --------------- | ------ | --- | --- | --- | --- |
|                              |     |     | θ                      | ψ ,thechallenge |        |     |     |     |     |
| istoidentifythedistributionP |     |     | inlatentspacewhichcon- |                 |        |     |     |     |     |
Growingsamplemodelsizeposesseveraladditionalchal-
| tainsthetargetedproperties. |     |     | Toapproximatethatdistribu- |     |     |     |     |     |     |
| --------------------------- | --- | --- | -------------------------- | --- | --- | --- | --- | --- | --- |
lenges,threeofwhichweaddresswiththefollowingmeth-
tion,previousworkusedalargenumberofwell-trainedmod-
ods. WeevaluatethesemethodsinAppendixA.
| els(Peeblesetal.,2022;Schu¨rholtetal.,2022a). |     |     |     | However, |     |     |     |     |     |
| --------------------------------------------- | --- | --- | --- | -------- | --- | --- | --- | --- | --- |
increasingthesizeofthesampledmodelsmakesgenerating Model Alignment. Symmetries in the weight space of
alargenumberofhigh-performancemodelsexceedinglyex- NNcomplicaterepresentationlearningoftheweights. The
pensive. Insteadofusingexpensivehigh-performancemod- numberofsymmetriesgrowsfastwithmodelsize(Bishop,
elstomodelP directly,weproposetofindaroughestimate 2006). Tomakerepresentationlearningeasier,wereduced
ofP,samplebroadly,andrefineP usingthesignalfromthe alltrainingmodelstoaunique,canonicalbasisofarefer-
sampledmodels. UsingE promptexamplesWe wecom- ence model. With reference model A we align model B
4

TowardsScalableandVersatileWeightSpaceLearning
by finding the permutation π = argmin ∥vec(Θ(A))− ImplementationDetails. Tomaintaindiversitywithineach
π
vec(Θ(B))∥2,whereΘ(A)aretheparametersofmodelA batch, we select only a single window from each model.
(Ainsworthetal.,2022). Wefixthesamereferencemodel Loading, preprocessing, and augmenting the entire sam-
acrossalldatasetsplitsandusethelastepochofeachmodel ple, only to use ca. 1% of it, is infeasible. To address
todeterminethepermutationforthatmodel. this, we leverage FFCV (Leclerc et al., 2023) to compile
datasetsconsistingofslicedandpermutedwindowsofmod-
Haloing. ThesequentialdecompositionofSANEdecouples
|     |     |     |     |     |     | els. Each | model | is super-sampled |     | for approximately |     | full |
| --- | --- | --- | --- | --- | --- | --------- | ----- | ---------------- | --- | ----------------- | --- | ---- |
thepretrainingsequencelengthfromdownstreamtaskse-
|                |     |                                     |     |     |     | coverage                      | within | the training | set, | considering         | the | ratio of |
| -------------- | --- | ----------------------------------- | --- | --- | --- | ----------------------------- | ------ | ------------ | ---- | ------------------- | --- | -------- |
| quencelengths. |     | Sincethememoryloadatinferenceiscon- |     |     |     |                               |        |              |      |                     |     |          |
|                |     |                                     |     |     |     | windowlengthtosequencelength. |        |              |      | FortheResNetzoos,we |     |          |
siderablylower,thesequencesatinferencecanbelonger.
|     |     |     |     |     |     | include140modelsperzoo, |     |     | anumberthatremainsman- |     |     |     |
| --- | --- | --- | --- | --- | --- | ----------------------- | --- | --- | ---------------------- | --- | --- | --- |
However,fullmodelsequencesmaystillnotfitinmemory
|                                  |     |     |     |                 |     | ageableintermsofmemoryandstorage. |     |     |     |     | Wetrainfor50 |     |
| -------------------------------- | --- | --- | --- | --------------- | --- | --------------------------------- | --- | --- | --- | --- | ------------ | --- |
| andmayhavetobeprocessedinslices. |     |     |     | Toensureconsis- |     |                                   |     |     |     |     |              |     |
epochsusingaOneCyclelearningratescheduler(Smith&
tencybetweentheslices,weaddcontextaroundthecontent
|          |                                           |     |     |     |     | Topin,2018). | Seedsarerecordedtoensurereproducibility. |     |     |     |     |     |
| -------- | ----------------------------------------- | --- | --- | --- | --- | ------------ | ---------------------------------------- | --- | --- | --- | --- | --- |
| windows. | Withaddedcontexthalobeforeandafterthecon- |     |     |     |     |              |                                          |     |     |     |     |     |
WebuildSANEinPyTorch(Paszkeetal.,2019),usingauto-
| tent window, | we  | get T | = T |     |     | .   |     |     |     |     |     |     |
| ------------ | --- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
hs,n n−h,..,n,...,n+ws,n+ws+h maticmixedprecisionandflashattention(Daoetal.,2022)
Similartoapproachesincomputervision(Vaswanietal.,
|     |     |     |     |     |     | toenhanceperformance. |     |     | Weuseray.tune(Liawetal.,2018) |     |     |     |
| --- | --- | --- | --- | --- | --- | --------------------- | --- | --- | ----------------------------- | --- | --- | --- |
2021), thiscontexthaloisaddedforthepassthroughen-
forhyperparameteroptimization.
coderanddecoder,butdisregardedafter.
Batch-NormConditioning. InmostcurrentNNmodels, 4.EmbeddingAnalysis
someparameterslikebatch-normweightsareupdateddur-
|             |        |         |         |            |       | In this section, |        | we analyze      | the | embeddings | of            | SANEand |
| ----------- | ------ | ------- | ------- | ---------- | ----- | ---------------- | ------ | --------------- | --- | ---------- | ------------- | ------- |
| ing forward | passes | instead | of with | gradients. | Since | that             |        |                 |     |            |               |         |
|             |        |         |         |            |       | compare          | to the | weight-analysis |     | methods    | WeightWatcher |         |
makesthemstructurallydifferent,weexcludetheseparame-
|     |     |     |     |     |     | (WW)(Martinetal.,2021). |     |     | Wefocusonthreeaspects: |     |     | i)  |
| --- | --- | --- | --- | --- | --- | ----------------------- | --- | --- | ---------------------- | --- | --- | --- |
tersfromrepresentationlearningandsamplingwithSANE.
globalrelationbetweenaccuracyandembeddings;ii)the
| Nonetheless, | theseparametersneedtobeinstantiatedfor |     |     |     |     |     |     |     |     |     |     |     |
| ------------ | -------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
trendofembeddingsoverlayerindex,asin(Martinetal.,
| sampledmodelstoworkwell. |     |     | Formodelsamplingmeth- |     |     |            |      |                    |     |             |        |       |
| ------------------------ | --- | --- | --------------------- | --- | --- | ---------- | ---- | ------------------ | --- | ----------- | ------ | ----- |
|                          |     |     |                       |     |     | 2021); and | iii) | the identification |     | of training | phases | as in |
ods,wethereforeproposetoconditionbatch-normparame-
(Martin&Mahoney,2019b;2021).
tersbyperformingafewforwardpasseswithsometarget
data. Importantly,thisprocessdoesnotupdatethelearned Toanalyzeweights,wefocusontwoWWmetricswhichin
previousworkrevealmodelperformanceaswellasinter-
weightsofthemodel.Itservestoalignthebatchnormstatis-
ticswiththemodel’sweights. nalmodelcomposition(correlationflow);thelogspectral
|     |     |     |     |     |     | norm log(∥W∥2 |     | ) and weighted |     | α, the | coefficient | of the |
| --- | --- | --- | --- | --- | --- | ------------- | --- | -------------- | --- | ------ | ----------- | ------ |
∞
3.TrainingSANE powerlawfittedtotheempiricalspectraldensity(Martin
|             |               |     |      |              |         | etal.,2021).                 | Thesetwometricsdescribedifferentaspects |     |                          |     |     |     |
| ----------- | ------------- | --- | ---- | ------------ | ------- | ---------------------------- | --------------------------------------- | --- | ------------------------ | --- | --- | --- |
|             |               |     |      |              |         | oftheeigenvaluedistribution. |                                         |     | Togetasimilarsignalonthe |     |     |     |
| We pretrain | SANEfollowing |     | Alg. | 1 on several | popula- |                              |                                         |     |                          |     |     |     |
tions of trained NN models, from the model zoo dataset internaldependencyofweightmatrices,wecomputeper-
|                          |     |     |                          |     |     | layerscalarszˆ |     | asthespreadofthetokensofonelayerin |     |     |     |     |
| ------------------------ | --- | --- | ------------------------ | --- | --- | -------------- | --- | ---------------------------------- | --- | --- | --- | --- |
| (Schu¨rholtetal.,2022c). |     |     | Weusezoosofsmallmodelsto |     |     |                | l   |                                    |     |     |     |     |
compare with previous work, as well as zoos containing hyper-representationspace,i.e.,theirstandarddeviation:
| largerResNet-18models. |     |     | Allzoosaresplitintotraining, |     |     |     |     |     |      |       |     |     |
| ---------------------- | --- | --- | ---------------------------- | --- | --- | --- | --- | --- | ---- | ----- | --- | --- |
|                        |     |     |                              |     |     |     |     | zˆ  | =std | (zt ) |     | (8) |
|                        |     |     |                              |     |     |     |     | l   |      | t m   |     |     |
validation,andtestsplits70:15:15.
|           |     |       |           |     |      |      |     | zt  | =g(Wt | ),  |     | (9) |
| --------- | --- | ----- | --------- | --- | ---- | ---- | --- | --- | ----- | --- | --- | --- |
|           |     |       |           |     |      |      |     | m   |       | m   |     |     |
| • Smaller | CNN | zoos. | The MNIST | and | SVHN | zoos |     |     |       |     |     |     |
wheregisthehyper-repencoder,zt
containLeNet-stylemodelswith3convolutionand2 arethestackedtokens
m
|                                    |     |     |     |     |             | toflayerm,andWt |     | istheweight-slicetoflayerm. |     |     |     |     |
| ---------------------------------- | --- | --- | --- | --- | ----------- | --------------- | --- | --------------------------- | --- | --- | --- | --- |
| denselayersandonly∼2.5kparameters. |     |     |     |     | Theslightly |                 |     | m                           |     |     |     |     |
largerCIFAR-10andSTL-10zoosusethesamearchi- TocompareWWmetricstoSANE,wepretrainSANEona
tecturewithwiderlayersand∼12kparameters.
Tiny-ImagenetResNet-18zooandcomputethetwometrics
• Larger ResNet zoos. We also use the CIFAR- on ResNets and VGGs of different sizes trained on Ima-
|     |            |     |               |      |            | geNetfrompytorchcv(Se´mery,2024). |     |     |     | OnbothResNetsin |     |     |
| --- | ---------- | --- | ------------- | ---- | ---------- | --------------------------------- | --- | --- | --- | --------------- | --- | --- |
| 10, | CIFAR-100, | and | Tiny-Imagenet | zoos | containing |                                   |     |     |     |                 |     |     |
ResNet-18 models (Schu¨rholt et al., 2022c) with ∼ Figure3,9andVGGsinFigure8,theWWmetricsandour
|     |                                               |     |     |     |     | embeddingsshowsimilarglobaltrends. |     |     |     |     | OnResNets,our |     |
| --- | --------------------------------------------- | --- | --- | --- | --- | ---------------------------------- | --- | --- | --- | --- | ------------- | --- |
| 12M | parameterstoevaluatescalabilitytolargemodels. |     |     |     |     |                                    |     |     |     |     |               |     |
embeddingsandWWhavelowvaluesatearlylayersand
Pretraining. WetrainSANEusingAlg. 1. Asaugmenta- asharpincreaseattheend. However,ourembeddingsadd
tions, we use noise and permutation. The permutation is an additional step for intermediate layers, which may in-
computed relative to the aligned model. For contrastive dicatethatSANEissensitivetoahigherdegreeofvariation
learning,thealignedmodelservesasoneview,andaper- intheselayerswhichpreviousworkfoundbycomparing
mutedversionasthesecondview. activations(Kornblithetal.,2019).
5

TowardsScalableandVersatileWeightSpaceLearning
5.EmpiricalPerformance
Inthissection,wedescribethegeneralperformanceSANE.
5.1.PredictingModelProperties
WeevaluateSANEfordiscriminativedownstreamtasksasa
|     |     |     |     |     |     |     | proxyforencodedmodelqualities. |     |     |     | Specifically,weinvesti- |     |     |
| --- | --- | --- | --- | --- | --- | --- | ------------------------------ | --- | --- | --- | ----------------------- | --- | --- |
gatewhetherSANEmatchesthepredictiveperformanceof
hyper-representationsonsmallCNNmodels(Table1)and
Figure3.Comparison between WeightWatcher (WW) features whethersimilarperformancecanbeachievedonResNet-
(left) and SANE(right). Features over layer index for ResNets 18models(Table2). Tothatend,wecomputemodelem-
frompytorchcvofdifferentsizes.
|     |     |     |     |     |     |     | beddings¯zasoutlinedinAlg. |     |     |                          | 2,andwecompareagainst |     |           |
| --- | --- | --- | --- | --- | --- | --- | -------------------------- | --- | --- | ------------------------ | --------------------- | --- | --------- |
|     |     |     |     |     |     |     | flattenedweightsW          |     |     | andweightstatisticss(W). |                       |     | Following |
In a second experiment, we aggregate the layer-wise em- theexperimentalsetupof(Eilertsenetal.,2020;Unterthiner
|            |                                            |     |     |     |     |     | et  | al., 2020; | Schu¨rholt | et al., | 2021), | we compute | embed- |
| ---------- | ------------------------------------------ | --- | --- | --- | --- | --- | --- | ---------- | ---------- | ------- | ------ | ---------- | ------ |
| beddingszˆ | l toevaluaterelationstomodelaccuracyinFig- |     |     |     |     |     |     |            |            |         |        |            |        |
dingsusingthethreemethodsandlinearprobefortestac-
ures4,10and11,similartopreviouswork(Martinetal.,
2021). OnmodelsfrompytorchcvandtheTiny-ImagNet curacy(Acc),epoch(Ep),andgeneralizationgap(Ggap).
Weagainusetrainedmodelsfromthemodelzoorepository
| model zoo | from | (Schu¨rholt | et al., | 2022c), | the WW | fea- |     |     |     |     |     |     |     |
| --------- | ---- | ----------- | ------- | ------- | ------ | ---- | --- | --- | --- | --- | --- | --- | --- |
turesandSANEembeddingsbothshowstrongcorrelations (Schu¨rholtetal.,2022c),withthesametrain,test,valsplits
asabove.
| to model | accuracy. | However, | while | the | WW metrics | are |     |     |     |     |     |     |     |
| -------- | --------- | -------- | ----- | --- | ---------- | --- | --- | --- | --- | --- | --- | --- | --- |
negativelycorrelatedtoaccuracy,ourembeddingsarepos- Table1. PropertypredictiononpopulationsofsmallCNNsused
| itivelycorrelatedtoaccuracy. |     |     | Thereasonforthatmaylie |     |     |     |     |     |     |     |     |     |     |
| ---------------------------- | --- | --- | ---------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
inpreviouswork(Schu¨rholtetal.,2021).Wereporttheregression
intheadditional’step’inFigure3. Thatis,largermodels R2onthetestsetpredictiontestaccuracyAcc.,epochEp.and
with more layers generally have higher performance. As generalizationgapGgapforlinearprobingwithmodelweights
W,modelweightsstatisticss(W)orSANEembeddingsasinputs.
Figure3shows,morelayersaddverysmallvaluesreducing
theglobalaverageforWWmetrics. Forourembeddings, MNIST SVHN CIFAR-10(CNN)
deepermodelshavemorelayerswithhigherzˆ values,due SANE SANE SANE
|     |     |     |     |     | l   |     |     | W   | S(W) | W   | S(W) |     | W S(W) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | ---- | --- | ------ |
totheafore-mentionedstep.Thisincreasestheglobalmodel ACC. 0.965 0.987 0.978 0.910 0.985 0.991 -7.580 0.965 0.885
averagewithgrowingmodelsize. Lastly,wecomparethe EP. 0.953 0.974 0.958 0.833 0.953 0.930 0.636 0.923 0.771
|                                 |     |     |     |                     |     |     | GGAP | 0.246 | 0.393 0.402 | 0.479 | 0.711 0.760 | 0.324 | 0.909 0.772 |
| ------------------------------- | --- | --- | --- | ------------------- | --- | --- | ---- | ----- | ----------- | ----- | ----------- | ----- | ----------- |
| eigenvaluespectrumtoembeddings. |     |     |     | Previousworkidenti- |     |     |      |       |             |       |             |       |             |
fieddistinctshapesatdifferenttrainingphasesorwithvary-
ingtraininghyperparameters(Martin&Mahoney,2019b; SANEmatchesbaselinesonsmallmodels. Theresultsof
|     |     |     |     |     |     |     | linear | probing | on small | CNNs | in Tables | 1   | and 9 confirm |
| --- | --- | --- | --- | --- | --- | --- | ------ | ------- | -------- | ---- | --------- | --- | ------------- |
2021). Whilewecanreplicatethedistributionsoftheeigen-
values,thedistributionsofourembeddingsonlyshowthe theperformanceofW (low)ands(W)(veryhigh)ofpre-
changefromearlyphasesoftrainingtotheheavy-taileddis- viouswork. SANEembeddingsshowcomparablyhighper-
formancetothes(W)andprevioushyper-representations.
tribution;seeFigure7.
AdditionalexperimentsinAppendixD.1comparetoprevi-
| In summary, |     | our embedding |     | analysis | indicates | that |                                 |     |     |     |     |                    |     |
| ----------- | --- | ------------- | --- | -------- | --------- | ---- | ------------------------------- | --- | --- | --- | --- | ------------------ | --- |
|             |     |               |     |          |           |      | ousworkandconfirmthesefindings. |     |     |     |     | Sequentialdecompo- |     |
SANErepresentsseveralaspectsofmodelquality(globally
sitionandrepresentationlearningaswellasusingthecen-
andonalayerlevel)thathavebeenestablishedpreviously.
terofgravitydoesnotsignificantlyreducetheinformation
containedinSANEembeddings.
|     |     |     |     |     |     |     | Table2. | Property | prediction |     | on ResNet-18 |     | model zoos of |
| --- | --- | --- | --- | --- | --- | --- | ------- | -------- | ---------- | --- | ------------ | --- | ------------- |
(Schu¨rholtetal.,2022c).WereporttheregressionR2onthetest
setpredictiontestaccuracyAcc.,epochEp.andgeneralization
gapGgapforlinearprobingwithmodelweightsstatisticss(W)
orSANEembeddingsasinputs.
|     |     |     |     |     |     |     |      | CIFAR-10 |       | CIFAR-100 |       | TINY-IMAGENET |       |
| --- | --- | --- | --- | --- | --- | --- | ---- | -------- | ----- | --------- | ----- | ------------- | ----- |
|     |     |     |     |     |     |     |      | S(W)     | SANE  | S(W)      | SANE  | S(W)          | SANE  |
|     |     |     |     |     |     |     | ACC. | 0.880    | 0.879 | 0.923     | 0.922 | 0.802         | 0.795 |
Figure4.ComparisonbetweenWeightWatcherfeatures(left)and
|              |          |     |            |          |             |     | EP.  | 0.999 | 0.999 | 0.999 | 0.992 | 0.999 | 0.980 |
| ------------ | -------- | --- | ---------- | -------- | ----------- | --- | ---- | ----- | ----- | ----- | ----- | ----- | ----- |
| SANE(right). | Accuracy |     | over model | features | for ResNets | and |      |       |       |       |       |       |       |
|              |          |     |            |          |             |     | GGAP | 0.490 | 0.512 | 0.882 | 0.879 | 0.704 | 0.699 |
AlthoughSANEispre-
VGGsfrompytorchcvofdifferentsizes.
trainedinaself-supervisedfashion,itpreservesthelinearrelation
ofaglobally-aggregatedembeddingtomodelaccuracy.
6

TowardsScalableandVersatileWeightSpaceLearning
SANEperformance prediction scales to ResNets. Both Table3. ModelgenerationonCNNmodelpopulationsfine-tuned
s(W) and SANEembeddings show similarly high perfor- onthesametask.WecomparetrainingfromscratchwithS
KDE30
from(Schu¨rholtetal.,2022a),SANEcombinedwiththeKDE30
| mance on | populations | of  | ResNet-18s; | see | Table | 2. On |     |     |     |     |     |     |     |     |
| -------- | ----------- | --- | ----------- | --- | ----- | ----- | --- | --- | --- | --- | --- | --- | --- | --- |
ResNet-18s, using the full weights W for linear prob- samplingmethod,andourSANEsubsampled.Eachofthesampled
ing is infeasible due to the size of the flattened weights. populationsisfine-tunedover25epochs.
SANEmatches the high performance of s(W). The re- Ep. Method MNIST SVHN CIFAR-10 STL
sultsshowthatsequentialhyper-representationsarecapa- ∼10/% ∼10/% ∼10/% ∼10/%
tr.fr.scratch
|                                                  |     |     |     |                      |     |     |     | S      |       | 68.6±6.7 | 54.5±5.9 |          | n/a | n/a      |
| ------------------------------------------------ | --- | --- | --- | -------------------- | --- | --- | --- | ------ | ----- | -------- | -------- | -------- | --- | -------- |
| bleofscalingtoResNet-18models.                   |     |     |     | Further,theaggrega-  |     |     |     | KDE30  |       |          |          |          |     |          |
|                                                  |     |     |     |                      |     |     |     | 0 SANE |       | 84.8±0.8 | 70.7±1.4 | 56.3±0.5 |     | 39.2±0.8 |
| tionevenoflongsequences(ca.                      |     |     |     | 50ktokens)embeddedin |     |     |     |        | KDE30 |          |          |          |     |          |
|                                                  |     |     |     |                      |     |     |     | SANE   |       | 86.7±0.8 | 72.3±1.6 | 57.9±0.2 |     | 43.5±1.0 |
| SANEpreservesmeaningfulinformationonmodelperfor- |     |     |     |                      |     |     |     |        | SUB   |          |          |          |     |          |
|                                                  |     |     |     |                      |     |     |     | SANE   |       | 20.8±0.1 | 21.6±0.5 | 19.3±0.2 |     | 17.5±1.5 |
GAUSS
mance,whichindicatesthefeasibilityofapplicationslike
|     |     |     |     |     |     |     |     | tr.fr.scratch |     | 20.6±1.6 | 19.4±0.6 | 37.2±1.4 |     | 21.3±1.6 |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | -------- | -------- | -------- | --- | -------- |
modeldiagnosticsortargetedsampling.
|     |     |     |     |     |     |     |     | S   |     | 83.7±1.3 | 69.9±1.6 |     | n/a | n/a |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | -------- | -------- | --- | --- | --- |
KDE30
|     |     |     |     |     |     |     |     | 1 SANE |     | 85.5±0.8 | 71.3±1.4 | 58.2±0.2 |     | 43.5±0.7 |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- | -------- | -------- | -------- | --- | -------- |
KDE30
5.2.GeneratingModels SANE 87.5±0.6 73.3±1.4 59.1±0.3 44.3±1.0
SUB
|     |     |     |     |     |     |     |     | SANE |     | 61.3±3.1 | 24.1±4.4 | 27.2±0.3 |     | 22.4±1.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | -------- | -------- | -------- | --- | -------- |
GAUSS
WeevaluateSANEforthegenerativedownstreamtasksi.e.,
|                          |     |     |                          |     |     |     |     | tr.fr.scratch |     | 36.7±5.2 | 23.5±4.7  | 48.5±1.0 |     | 31.6±4.2 |
| ------------------------ | --- | --- | ------------------------ | --- | --- | --- | --- | ------------- | --- | -------- | --------- | -------- | --- | -------- |
| forsamplingmodelweights. |     |     | Wegenerateweightsfollow- |     |     |     |     |               |     |          |           |          |     |          |
|                          |     |     |                          |     |     |     |     | S             |     | 92.4±0.7 | 57.3±12.4 |          | n/a | n/a      |
KDE30
ingAlg. 3andtesttheminfine-tuning, transferlearning, 5 SANE 87.5±0.7 72.2±1.2 58.8±0.4 45.2±0.6
KDE30
andhowtheygeneralizetonewtasksandarchitectures. In SANE 89.0±0.4 73.6±1.5 59.6±0.3 45.3±0.9
SUB
the following paragraphs, we begin with experiments on SANE 83.4±0.8 35.6±8.9 43.3±0.3 34.2±0.7
GAUSS
smallCNNmodelsfromthemodelzoorepositorytocom- tr.fr.scratch 83.3±2.6 66.7±8.5 57.2±0.8 44.0±1.0
parewithpreviouswork(Tables3,13). Subsequently,we S 93.0±0.7 74.2±1.4 n/a n/a
KDE30
|                                                    |     |     |     |                 |     |     | 25  | SANE          |       | 92.0±0.3 | 74.7±0.8  | 60.2±0.6 |     | 48.4±0.5 |
| -------------------------------------------------- | --- | --- | --- | --------------- | --- | --- | --- | ------------- | ----- | -------- | --------- | -------- | --- | -------- |
| evaluateSANEforsamplingResNet-18modelsforfinetun-  |     |     |     |                 |     |     |     |               | KDE30 |          |           |          |     |          |
|                                                    |     |     |     |                 |     |     |     | SANE          |       | 92.3±0.4 | 75.1±1.0  | 61.2±0.1 |     | 48.0±0.4 |
| ingandtransferlearning(Tables4,14).                |     |     |     | Lastly,weevalu- |     |     |     |               | SUB   |          |           |          |     |          |
|                                                    |     |     |     |                 |     |     |     | SANE          |       | 94.2±0.4 | 54.2±17.6 | 52.2±0.6 |     | 43.5±0.5 |
| atesamplingfornewtasksandnewarchitecturesusingonly |     |     |     |                 |     |     |     |               | GAUSS |          |           |          |     |          |
|                                                    |     |     |     |                 |     |     | 50  | tr.fr.scratch |       | 91.1±2.6 | 70.7±8.8  | 61.5±0.7 |     | 47.4±0.9 |
fewpromptexamples(Figure5andTables5,15,16,17).
WepretrainSANEonmodelsfromthefirsthalfofthetrain-
| ingepochswithAlg. |     | 1,andkeeptheremainingepochs(26- |     |     |     |     |       |      |      |            |             |     |              |     |
| ----------------- | --- | ------------------------------- | --- | --- | --- | --- | ----- | ---- | ---- | ---------- | ----------- | --- | ------------ | --- |
|                   |     |                                 |     |     |     |     | small | CNNs | that | sequential | pretraining |     | and sampling | of  |
50)asholdouttocompareagainst,followingtheexperimen-
|                                    |     |     |     |                   |     |     | SANEimprovesperformance,particularlyzeroshot. |     |     |     |     |     |     | This |
| ---------------------------------- | --- | --- | --- | ----------------- | --- | --- | --------------------------------------------- | --- | --- | --- | --- | --- | --- | ---- |
| talsetupof(Schu¨rholtetal.,2022a). |     |     |     | WesampleusingAlg. |     |     |                                               |     |     |     |     |     |     |      |
indicatesthepotentialforscenarioswithlittlelabelleddata.
3andusemodelsfromthelastepochinthepretrainingset
(epoch 25) as prompt examples. We denote subsampling SANESequential Sampling Scales to ResNets. To eval-
SANEscales
withSANE SUB anditerativelyupdatingthedistributionP uate how well sampling with to larger mod-
as SANE . To evaluate the impact of the sampling els,wecontinuewithexperimentsonResNet-18s. There-
BOOT
method,wealsocombineSANEwiththeKDE30sampling sults of these experiments Tables 4 and 14 show that de-
approachthatuseshigh-qualitypromptexamples(Schu¨rholt spitethelongsequences,thesampledResNetmodelsper-
etal.,2022a). Wefurtherevaluatesamplingwithoutprompt formwellaboverandominitialization. Forexample,sam-
examplesbybootstrappingoffofaGaussianpriorP, de- pledResNet-18sachieve68.1%onCIFAR-10withoutany
notedasSANE . Wecompareagainsttrainingfrom fine-tuning(Table4). Thesemodelsareatleastthreeorders
GAUSS
scratch,aswellasfine-tuningfromthepromptexamples. ofmagnitudelargerthanpreviousmodelsusedforhyper-
representationlearning(Schu¨rholtetal.,2022a),rendering
| Sampling | High-Performing |     | CNNs | Zero-Shot. |     | We be- |     |     |     |     |     |     |     |     |
| -------- | --------------- | --- | ---- | ---------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
itcomputationallyinfeasiblefortheapproachpresentedin
| gin with                     | finetuning | and transfer  |     | learning experiments |          | on   |                                  |     |                |            |       |                     |                |        |
| ---------------------------- | ---------- | ------------- | --- | -------------------- | -------- | ---- | -------------------------------- | --- | -------------- | ---------- | ----- | ------------------- | -------------- | ------ |
|                              |            |               |     |                      |          |      | (Schu¨rholt                      |     | et al., 2022a) |            | to be | evaluated           | against.       | As be- |
| small CNNs                   | from       | the modelzoo  |     | dataset to           | validate | that |                                  |     |                |            |       |                     |                |        |
|                              |            |               |     |                      |          |      | fore,                            | the | performance    | difference |       | to random           | initialization |        |
| the sequential               |            | decomposition | for | pretraining          | and      | sam- |                                  |     |                |            |       |                     |                |        |
|                              |            |               |     |                      |          |      | becomessmallerduringfine-tuning. |     |                |            |       | Similartoourexperi- |                |        |
| plingdoesnothurtperformance. |            |               |     | Theresultsoftheseex- |          |      |                                  |     |                |            |       |                     |                |        |
mentsonCNNs,sampledResNet-18sachievecompetitive
perimentsshowdramaticallyimprovedperformancezero-
performanceorevenoutperformtrainingfromscratchwith
| shot for               | fine-tuning | and  | transfer | learning              | over previous |       |                                           |       |               |        |     |            |             |      |
| ---------------------- | ----------- | ---- | -------- | --------------------- | ------------- | ----- | ----------------------------------------- | ----- | ------------- | ------ | --- | ---------- | ----------- | ---- |
|                        |             |      |          |                       |               |       | aconsiderablysmallercomputationalbudget.1 |       |               |        |     |            | Transferred |      |
| hyper-representations; |             | see  | Tables   | 3 and 13.             | At            | epoch |                                           |       |               |        |     |            |             |      |
|                        |             |      |          |                       |               |       | to                                        | a new | task, sampled | models |     | outperform | training    | from |
| 0, SANEimproves        |             | over | previous | hyper-representations |               |       |                                           |       |               |        |     |            |             |      |
scratchandmatchfine-tuningfrompromptexamples(Table
| S KDE30          | byalmost20%. |              | Theeffectbecomessmallerdur- |     |     |         |      |                                                   |     |     |     |     |     |     |
| ---------------- | ------------ | ------------ | --------------------------- | --- | --- | ------- | ---- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|                  |              |              |                             |     |     |         | 14). | Interestingly,subsamplingandbootstrappingappearto |     |     |     |     |     |     |
| ing fine-tuning. |              | Nonetheless, | SANEconsistently            |     |     | outper- |      |                                                   |     |     |     |     |     |     |
forms training from scratch with a higher epoch budget, 1Thebasepopulationistrainedwithaone-cyclelearningrate
oftenbyseveralpercentagepoints. Thisdemonstrateson scheduler. Toavoidanybias,weadoptthesameschedulerbut
trainforonly10epochs,whichaffectsdirectcomparability.
7

TowardsScalableandVersatileWeightSpaceLearning
Table4. ModelgenerationonResNet-18modelpopulationsfine- MNIST.Theseresultsshowthatoursamplingmethodsnot
tunedonthesametask.Wecomparesampledmodelsatdifferent onlydroprequirementsforthepromptexamplesbuteven
epochswithmodelstrainedfromscratch.
improvetheperformanceofthesampledmodels.
| Epoch |     | Method | CIFAR-10 | CIFAR-100 |     | Tiny-Imagenet |     |     |     |     |     |     |
| ----- | --- | ------ | -------- | --------- | --- | ------------- | --- | --- | --- | --- | --- | --- |
Few-ShotModelSamplingTransferstoNewTasksand
|     | tr.fr.scratch |       | ∼10/%    | ∼1/%     |     | ∼0.5/%   |     |                                                    |                                     |     |     |               |
| --- | ------------- | ----- | -------- | -------- | --- | -------- | --- | -------------------------------------------------- | ----------------------------------- | --- | --- | ------------- |
|     | 0             |       |          |          |     |          |     | Architectures.                                     | Lastly,weexplorewhethersamplingmod- |     |     |               |
|     | SANE          |       | 64.8±2.0 | 19.8±2.5 |     | 8.4±0.9  |     |                                                    |                                     |     |     |               |
|     |               | KDE30 |          |          |     |          |     | elsusingSANEgeneralizesbeyondtheoriginaltaskandar- |                                     |     |     |               |
|     | SANE          |       | 68.1±0.7 | 19.8±1.3 |     | 11.1±0.5 |     |                                                    |                                     |     |     |               |
|     |               | SUB   |          |          |     |          |     | chitecturewithveryfewpromptexamples.               |                                     |     |     | Suchtransfers |
|     | SANE          |       | 68.6±1.2 | 20.4±1.3 |     | 11.7±0.5 |     |                                                    |                                     |     |     |               |
BOOT
areoutofreachofprevioushyper-representations,which
|     | tr.fr.scratch |     | 43.7±1.3 | 17.5±0.7 |     | 13.8±0.8 |     |                                  |     |     |                 |     |
| --- | ------------- | --- | -------- | -------- | --- | -------- | --- | -------------------------------- | --- | --- | --------------- | --- |
|     | 1             |     |          |          |     |          |     | areboundtoafixednumberofweights. |     |     | SANE,ontheother |     |
|     | SANE          |     | 82.4±0.9 | 59.0±1.3 |     | 46.7±0.8 |     |                                  |     |     |                 |     |
KDE30
SANE 83.6±1.5 60.8±0.8 47.4±1.0 hand,representsmodelsofdifferentsizesorarchitectures
SUB
SANE 82.8±1.4 60.2±0.5 47.2±0.8 simplyassequencesofdifferentlengths,whichcanvarybe-
BOOT
|     |               |     |          |          |     |          |     | tweenpretrainingandsampling. |     | Sinceweusetheprompt |     |     |
| --- | ------------- | --- | -------- | -------- | --- | -------- | --- | ---------------------------- | --- | ------------------- | --- | --- |
|     | tr.fr.scratch |     | 64.4±2.9 | 36.5±2.0 |     | 31.1±1.6 |     |                              |     |                     |     |     |
5
SANE 85.9±0.6 56.2±1.7 45.6±1.4 examplesonlytoroughlymodelthesamplingdistribution,
KDE30
SANE 85.4±1.3 56.7±1.6 45.7±0.8 weneedonlyafew(1-5)promptexampleswhicharetrained
SUB
|     | SANE |      | 85.4±0.7 | 56.4±1.2 |     | 49.1±1.7 |     |                         |     |          |                |     |
| --- | ---- | ---- | -------- | -------- | --- | -------- | --- | ----------------------- | --- | -------- | -------------- | --- |
|     |      | BOOT |          |          |     |          |     | foronlyafewepochs(1-5). |     | Thatway, | samplingfornew |     |
tr.fr.scratch 76.5±2.7 49.0±2.0 39.9±2.2 architectures and/or tasks can become very efficient. We
10
SANE 91.4±0.1 72.9±0.2 64.2±0.3 testthatideainthreeexperiments: (i)changingthetasksbe-
KDE30
|     | SANE |      | 91.6±0.2 | 72.9±0.1 |     | 64.0±0.2 |     |                                                     |     |     |     |     |
| --- | ---- | ---- | -------- | -------- | --- | -------- | --- | --------------------------------------------------- | --- | --- | --- | --- |
|     |      | SUB  |          |          |     |          |     | tweenpretrainingandprompt-examplesfromCIFAR-100     |     |     |     |     |
|     | SANE |      | 91.6±0.2 | 72.8±0.1 |     | 64.1±0.2 |     |                                                     |     |     |     |     |
|     |      | BOOT |          |          |     |          |     | toTiny-Imagenet(Table5);(ii)changingthearchitecture |     |     |     |     |
25 tr.fr.scratch 85.5±1.5 56.5±2.0 43.3±1.9 betweenpretrainingandprompt-examplesfromResNet-18
| 50  | tr.fr.scratch |     | 92.14±0.2 | 70.7±0.4 |     | 57.3±0.6 |     |     |     |     |     |     |
| --- | ------------- | --- | --------- | -------- | --- | -------- | --- | --- | --- | --- | --- | --- |
toResNet-34(Table15);and(iii)changingbothtaskand
| 60  | tr.fr.scratch |     | n/a | 74.2±0.3 |     | 63.9±0.5 |     |     |     |     |     |     |
| --- | ------------- | --- | --- | -------- | --- | -------- | --- | --- | --- | --- | --- | --- |
architecturefromResNet-18onCIFAR-100toResNet-34
onTiny-Imagenet(Figure5andTable16).
workwellwhenthereisausefulsignaltostartwith,i.e.,on Inallthreeexperiments,usingtargetpromptexamplesim-
easiertasksthataresimilartothepretrainingdistribution.
provesoverrandominitializationaswellasprevioustransfer
Thissuggeststhatthesamplingdistributionsarenotideal, experiments. ThisindicatesthatSANErepresentationscon-
andmayrequireabetterfit,moresamples,oriterativead- tainusefulinformationevenfornewarchitecturesortasks.
| justmenttofitnewdatasetszero-shot. |     |     |     |     | Nonetheless,eventhe |     |     |     |     |     |     |     |
| ---------------------------------- | --- | --- | --- | --- | ------------------- | --- | --- | --- | --- | --- | --- | --- |
Thesampledmodelsoutperformthepromptexamplesand
relativelynaivesamplingmethodscansuccessfullysample trainingfromscratch,considerablyinearlierepochs,and
| competitivemodels, |     |     | evenatthescaleofResNet-sizedar- |     |     |     |     |     |     |     |     |     |
| ------------------ | --- | --- | ------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
preserveaperformanceadvantagethroughoutfine-tining.
| chitectures. |     | Thisshowsthatoursequentialsamplingworks |     |     |     |     |     |          |       |     |     |     |
| ------------ | --- | --------------------------------------- | --- | --- | --- | --- | --- | -------- | ----- | --- | --- | --- |
|              |     |                                         |     |     |     |     |     | Sampling | for a |     |     |     |
evenforlongsequencesoftokens.
|     |     |     |     |     |     |     |     | new task | (Table 5), |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | --- | --- | --- |
Table5. SamplingResNet-18models
| SubsamplingImprovesPerformance.                       |             |     |                 |     | Previousworkre- |         |     | the sampled       | mod- |                    |              |              |
| ----------------------------------------------------- | ----------- | --- | --------------- | --- | --------------- | ------- | --- | ----------------- | ---- | ------------------ | ------------ | ------------ |
|                                                       |             |     |                 |     |                 |         |     |                   |      | for Tiny-Imagenet. |              | SANEwas pre- |
| quireshigh-qualitypromptexamplestotargetspecificprop- |             |     |                 |     |                 |         |     | els outperform    | the  |                    |              |              |
|                                                       |             |     |                 |     |                 |         |     |                   |      | trained on         | CIFAR-100,   | 15 samples   |
| erties                                                | (Schu¨rholt |     | et al., 2022a). | Our | sampling        | methods |     |                   |      |                    |              |              |
|                                                       |             |     |                 |     |                 |         |     | promptexamplesaf- |      | are drawn using    | subsampling, | and 5        |
droptheserequirementsandusepromptexamplesonlyto ter just two epochs prompt examples are taken from the
modelaprior. WethereforecompareSANEwithS Tiny-ImagenetResNet-18zooatepoch
|     |     |     |     |     |     |     | KDE30 | of  | fine-tuning, |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----- | --- | ------------ | --- | --- | --- |
from(Schu¨rholtetal.,2022a)toSANE.Further,wecom-
|     |     |     |     |     |     |     |     | whichindicatesthat |     | 25withameanaccuracyof43%. |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | ------------------------- | --- | --- |
paretheKDE30samplingmethodwithoursubsampling transfer-learning ResNet-18CIFAR100toTinyImagnet
| approach      |     | on SANE.                          | On datasets | where | published |     | results |              |          |               |     |         |
| ------------- | --- | --------------------------------- | ----------- | ----- | --------- | --- | ------- | ------------ | -------- | ------------- | --- | ------- |
|               |     |                                   |             |       |           |     |         | using SANEis | an       | Ep. Method    |     | AccTI   |
| areavailable, |     | usingKDE30withSANEimprovesperfor- |             |       |           |     |         | efficient    | alterna- |               |     |         |
|               |     |                                   |             |       |           |     |         |              |          | tr.fr.scratch |     | 0.5±0.0 |
manceoverpreviouslypublishedresultswithS ;see tive. Sampling 0
|                                           |     |     |     |     |     | KDE30 |     |                |     | SANE          |     | 0.6±0.0  |
| ----------------------------------------- | --- | --- | --- | --- | --- | ----- | --- | -------------- | --- | ------------- | --- | -------- |
| Table3forMNISTandSVHNresults,e.g.,epoch0. |     |     |     |     |     |       | We  | from ResNet-18 | to  |               |     |          |
|                                           |     |     |     |     |     |       |     |                |     | tr.fr.scratch |     | 10.4±2.2 |
creditthistothebetterreconstructionqualityofpre-training ResNet-34 for the 1
|     |     |     |     |     |     |     |     |     |     | SANE |     | 39.4±1.5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | -------- |
withSANE.Further,oursamplingmethodsimproveperfor-
|     |     |     |     |     |     |     |     | same task | (Table |               |     |          |
| --- | --- | --- | --- | --- | --- | --- | --- | --------- | ------ | ------------- | --- | -------- |
|     |     |     |     |     |     |     |     |           |        | tr.fr.scratch |     | 28.5±0.9 |
manceoverS KDE30 . WecompareSANE+S KDE30 with 15) shows likewise 2
|                                                  |     |     |     |     |     |     |     |          |         | SANE |     | 61.0±0.2 |
| ------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | -------- | ------- | ---- | --- | -------- |
| SANE+subsamplingandSANE+bootstrapping,e.g.,inTa- |     |     |     |     |     |     |     | improved | perfor- |      |     |          |
ble4onCIFAR-10atepoch0from64.8%to68.1%,oron 2 SANEensamble 64.0
manceovertraining
| TinyImagenetfrom8.4%to11.1%. |     |                                            |     |     | Usingbootstrapping |     |     | fromscratch,which |     |     |     |     |
| ---------------------------- | --- | ------------------------------------------ | --- | --- | ------------------ | --- | --- | ----------------- | --- | --- | --- | --- |
| toadjustP                    |     | iterativelyfurtherimprovesthesampledmodels |     |     |                    |     |     |                   |     |     |     |     |
indicatesthatthelearnedrepresentationgeneralizestolarger
slightly. Itevenallowstoreplacepromptexampleswitha architecturesaswell. Samplingfornewtasksanddifferent
GaussianpriorP. TheresultsofSANE showhigh architecture(Figure5andTable16)combinestheprevious
GAUSS
performanceafterfine-tuning,eventhehighestoverallon
8

TowardsScalableandVersatileWeightSpaceLearning
|     |     |     |     |     | (Ashkenazietal.,2022;DeLuigietal.,2023). |     |     | Otherwork |     |
| --- | --- | --- | --- | --- | ---------------------------------------- | --- | --- | --------- | --- |
investigatesthestructureoftrainedweightsonafundamen-
tallevel,usingtheireigenorsingularvaluedecompositions
toidentifytrainingphasesorpredictproperties(Martin&
Mahoney,2019b;2020;Martinetal.,2021;Martin&Ma-
honey,2021;Yangetal.,2022;Meller&Berkouk,2023).
Takinganoptimizationperspective,otherworkhasinvesti-
gatedtheuniquenessofthebasisoftrainedNNs(Ainsworth
|     |     |     |     |     | etal.,2022;Brownetal.,2023). |               |               | Otherworkidentifiessub- |     |
| --- | --- | --- | --- | --- | ---------------------------- | ------------- | ------------- | ----------------------- | --- |
|     |     |     |     |     | spaces of                    | weights that  | are relevant, | which motivates         | our |
|     |     |     |     |     | work (Benton                 | et al., 2021; | Lucas         | et al., 2021; Wortsman  |     |
Figure5.Comparisonbetweensampledmodelsandrandomini-
|     |     |     |     |     | etal.,2021;Fort&Jastrzebski,2019). |     |     | Themodeconnec- |     |
| --- | --- | --- | --- | --- | ---------------------------------- | --- | --- | -------------- | --- |
tializationtrainedfor5epochsTiny-Imagenet.Differentarchitec-
turesaresampledfromSANEpretrainedonaResNet-18CIFAR- tivity of trained modelshas been investigatedto improve
100zoo. Althoughbothmodelsandtasksarechanged,sampled understandingofhowtotrainmodels(Draxleretal.,2018;
| modelsperformbetter. |     |     |     |     | Nguyen,2019;Frankleetal.,2019). |     |     |     |     |
| -------------------- | --- | --- | --- | --- | ------------------------------- | --- | --- | --- | --- |
Adifferentlineofworktrainsmodelstogenerateweights
fortargetmodels,suchasHyperNetworks(Haetal.,2016;
| experiments | and confirms | their | results. Sampled | models |     |     |     |     |     |
| ----------- | ------------ | ----- | ---------------- | ------ | --- | --- | --- | --- | --- |
outperformtrainingfromscratchbyaconsiderablemargin. Nguyen et al., 2019; Zhang et al., 2019; Knyazev et al.,
|          |           |                      |          |          | 2021; 2023; | Kofinas et | al., 2024), | with a recurrent | back- |
| -------- | --------- | -------------------- | -------- | -------- | ----------- | ---------- | ----------- | ---------------- | ----- |
| Figure 5 | indicates | that with increasing | distance | from the |             |            |             |                  |       |
bone(Wangetal.,2023)aslearnedinitialization(Dauphin
pretrainingarchitectureforSANE,theperformancegainof
sampledmodelsdecreases,e.g.,withincreasingResNetsize. &Schoenholz,2019)orformetalearning(Finnetal.,2017;
|     |     |     |     |     | Zhmoginovetal.,2022;Navaetal.,2022). |     |     | Whilethelast |     |
| --- | --- | --- | --- | --- | ------------------------------------ | --- | --- | ------------ | --- |
Additionally,sincesamplingmodelsusingSANEischeap
andlendsitselftoensembling,weinvestigatethediversityof categoryusesdatatogetlearningsignals,anotherlineof
|                             |     |     |                         |     | worklearnsrepresentationsoftheweightsdirectly. |     |     |     | Hyper- |
| --------------------------- | --- | --- | ----------------------- | --- | ---------------------------------------------- | --- | --- | --- | ------ |
| sampledmodelsinAppendixE.1. |     |     | Takentogether,theexper- |     |                                                |     |     |     |        |
Representationstrainanencoder-decoderarchitectureusing
imentsshowthatSANElearnsrepresentationsthatcangen-
eralizebeyondthepretrainingtaskandarchitecture,andcan reconstructionoftheweights,withcontrastiveguidance,and
hasbeenproposedtopredictmodelproperties(Schu¨rholt
efficientlybesampledforbothnewtasksandarchitectures.
|     |     |     |     |     | et al., 2021) | or generate                            | new models | (Schu¨rholt | et al., |
| --- | --- | --- | --- | --- | ------------- | -------------------------------------- | ---------- | ----------- | ------- |
|     |     |     |     |     | 2022b;a).     | Whilepreviousworkwaslimitedtosmallmod- |            |             |         |
Limitations
elsoffixedlength,thispaperproposesmethodstodecouple
Inthispaper,wepretrainSANEonhomogeneouszooswith
|     |     |     |     |     | therepresentationlearnersizefromthebasemodel. |     |     |     | Related |
| --- | --- | --- | --- | --- | --------------------------------------------- | --- | --- | --- | ------- |
onearchitecture. Thissimplifiesalignmentforpre-training, approachesuseconvolutionalauto-encoders(Berardietal.,
| but more | importantly | it simplifies | evaluation | for model |     |     |     |     |     |
| -------- | ----------- | ------------- | ---------- | --------- | --- | --- | --- | --- | --- |
2022)ordiffusionontheweights(Peeblesetal.,2022).
SinceSANEcantrainonvaryingarchitectures
generation.
andmodelsizes,themodelpopulationrequirementforpre-
7.Conclusion
trainingissignificantlyrelaxed.Asufficientnumberofmod-
elsareavailableonpublicmodelhubs. Further, oursam- In this work, we propose SANE, a method to learn
plingmethodrequiresaccesstopromptexamples,tohave
|     |     |     |     |     | task-agnostic | representations | of  | Neural Network | mod- |
| --- | --- | --- | --- | --- | ------------- | --------------- | --- | -------------- | ---- |
aninformedpriorfromwhichtosample. Forsmallmodels, els. SANEdecouples model tokenization from hyper-
bootstrappingfromaGaussianfindsthetargeteddistribu- representationlearningandcanscaletomuchlargerneural
| tion; seeSANE |     | inTable3. | Forlargemodelswith |     |     |     |     |     |     |
| ------------- | --- | --------- | ------------------ | --- | --- | --- | --- | --- | --- |
GAUSS network models and generalize to models of different ar-
correspondinglylongsequences,thatapproachistooexpen- chitectures. WeanalyzeSANEembeddingsandfindtheyre-
sive,whichiswhywerelyonpromptexamples. Lastly,in vealmodelqualitymetrics. Empiricalevaluationsshowthat
thispaper,weperformexperimentsonlyoncomputervision i)SANEembeddingscontaininformationonmodelquality
tasks. Thisisachoicetosimplifytheexperimentsetup. bothgloballyandonalayerlevel,ii)SANEembeddingsare
predictiveofmodelperformance,andiii)samplingmodels
6.RelatedWork with SANEachieves higher performance and generalizes
|     |     |     |     |     | tolargermodelsandnewarchitectures. |     |     | Further,wepropose |     |
| --- | --- | --- | --- | --- | ---------------------------------- | --- | --- | ----------------- | --- |
RepresentationlearninginthespaceofNNweightshasbe-
samplingmethodsthatreducequalityandquantityrequire-
comeagrowingfieldrecently. Severalmethodswithdiffer- mentsforpromptexamplesandallowtargetingnewmodel
entapproachestodealwithweightspaceshavebeenpro-
distributions.
| posed to | predict model | properties | such as accuracy | (Un- |     |     |     |     |     |
| -------- | ------------- | ---------- | ---------------- | ---- | --- | --- | --- | --- | --- |
terthineretal.,2020;Eilertsenetal.,2020;Andreisetal.,
2023;Zhangetal.,2023)ortolearntheencodedconcepts
9

TowardsScalableandVersatileWeightSpaceLearning
Acknowledgements Dao,T.,Fu,D.Y.,Ermon,S.,Rudra,A.,andRe´,C.FlashAt-
|     |     |     |     | tention: | FastandMemory-EfficientExactAttentionwith |     |     |
| --- | --- | --- | --- | -------- | ----------------------------------------- | --- | --- |
KSandDBwouldliketoacknowledgetheGoogleResearch
IO-Awareness,June2022.
ScholarAwardandtheSwissNationalScienceFoundation
for partial funding of this work. We are thankful to Erik Dauphin,Y.N.andSchoenholz,S. MetaInit: Initializing
VeeatGoogleResearchfortheinsightfuldiscussions,Le´o learningbylearningtoinitialize. InNeuralInformation
MeynentandJoelleHannaforeditorialsupport,andHar- ProcessingSystems,2019.
aldRotter,KurtSta¨dler,andMartinEigenmannforthesup-
DeLuigi,L.,Cardace,A.,Spezialetti,R.,Ramirez,P.Z.,
portwiththecomputationalinfrastructurewhichmadethis
|     |     |     |     | Salti,S.,andDiStefano,L. |     | DeepLearningonImplicit |     |
| --- | --- | --- | --- | ------------------------ | --- | ---------------------- | --- |
projectpossible.MWMwouldliketoacknowledgetheNSF,
NeuralRepresentationsofShapes,February2023.
DOE,andIARPAforpartialsupportofthiswork.
|                 |     |     |     | Deutsch,L. | GeneratingNeuralNetworkswithNeuralNet- |     |     |
| --------------- | --- | --- | --- | ---------- | -------------------------------------- | --- | --- |
| ImpactStatement |     |     |     | works.     | April2018.                             |     |     |
Thispaperintroducesanovelweightrepresentationlearning
Draxler,F.,Veschgini,K.,Salmhofer,M.,andHamprecht,
methoddesignedtoenhancetheperformanceandscalability F. Essentially No Barriers in Neural Network Energy
| ofmachinelearningmodelsacrossvariousapplications. |     |     | As  |            |                  |            |            |
| ------------------------------------------------- | --- | --- | --- | ---------- | ---------------- | ---------- | ---------- |
|                                                   |     |     |     | Landscape. | In International | Conference | on Machine |
afundamentalapproach,itservesasafoundationforfuture Learning,March2018.
| advancementsinthefieldofmachinelearning. |     |     | SANEholds |     |     |     |     |
| ---------------------------------------- | --- | --- | --------- | --- | --- | --- | --- |
potential for use in both academic research and industry Eilertsen, G., Jo¨nsson, D., Ropinski, T., Unger, J., and
|     |     |     |     | Ynnerman,A. | Classifyingtheclassifier: |     | Dissectingthe |
| --- | --- | --- | --- | ----------- | ------------------------- | --- | ------------- |
applicationsandthusinheritsalltheirbenefitsbutalsorisks
foradverseapplicationsofmachinelearning. Itsversatility weightspaceofneuralnetworks. February2020.
SANEa
and scalability make valuable tool that may also Finn,C.,Abbeel,P.,andLevine,S. Model-AgnosticMeta-
offerinsightsintomodelinterpretability.
|     |     |     |     | LearningforFastAdaptationofDeepNetworks. |                           |            | InPro- |
| --- | --- | --- | --- | ---------------------------------------- | ------------------------- | ---------- | ------ |
|     |     |     |     | ceedings                                 | of the 34th International | Conference | on Ma- |
References
chineLearning.PMLR,July2017.
Ainsworth, S. K., Hayase, J., and Srinivasa, S. Git Re- Fort,S.andJastrzebski,S. LargeScaleStructureofNeural
Basin:MergingModelsmoduloPermutationSymmetries, NetworkLossLandscapes. June2019.
September2022.
Frankle,J.,Dziugaite,G.,Roy,D.M.,andCarbin,M.Linear
Andreis, B., Bedionita, S., and Hwang, S. J. Set-based Mode Connectivity and the Lottery Ticket Hypothesis.
December2019.
NeuralNetworkEncoding,May2023.
|     |     |     |     | Ha,D.,Dai,A.,andLe,Q.V. |     | HyperNetworks,2016. |     |
| --- | --- | --- | --- | ----------------------- | --- | ------------------- | --- |
Ashkenazi,M.,Rimon,Z.,Vainshtein,R.,Levi,S.,Richard-
| son, E., Mintz, | P., and Treister, | E. NeRN | – Learning |                                             |     |     |          |
| --------------- | ----------------- | ------- | ---------- | ------------------------------------------- | --- | --- | -------- |
|                 |                   |         |            | Jiang,Y.,Krishnan,D.,Mobahi,H.,andBengio,S. |     |     | Predict- |
NeuralRepresentationsforNeuralNetworks,December ingtheGeneralizationGapinDeepNetworkswithMar-
2022.
|     |     |     |     | ginDistributions. | June2019. |     |     |
| --- | --- | --- | --- | ----------------- | --------- | --- | --- |
Benton,G.W.,Maddox,W.J.,Lotfi,S.,andWilson,A.G.
|     |     |     |     | Knyazev, | B., Drozdzal, | M., Taylor, G. | W., and Romero- |
| --- | --- | --- | --- | -------- | ------------- | -------------- | --------------- |
LossSurfaceSimplexesforModeConnectingVolumes Soriano,A. ParameterPredictionforUnseenDeepAr-
andFastEnsembling. InPMLR,2021. chitectures. InConferenceonNeuralInformationPro-
cessingSystems(NeurIPS),2021.
Berardi,G.,DeLuigi,L.,Salti,S.,andDiStefano,L.Learn-
ingtheSpaceofDeepModels,June2022. Knyazev, B., Hwang, D., andLacoste-Julien, S. CanWe
ScaleTransformerstoPredictParametersofDiverseIma-
Bishop,C.M. PatternRecognitionandMachineLearning. geNetModels? InarXiv.Org,March2023.
springer,2006.
Kofinas,M.,Knyazev,B.,Zhang,Y.,Chen,Y.,Burghouts,
| Brown, D., Vyas, | N., and Bansal, | Y. On Privileged | and |                |            |           |                  |
| ---------------- | --------------- | ---------------- | --- | -------------- | ---------- | --------- | ---------------- |
|                  |                 |                  |     | G. J., Gavves, | E., Snoek, | C. G. M., | and Zhang, D. W. |
Convergent Bases in Neural Network Representations, Graphneuralnetworksforlearningequivariantrepresen-
| July2023. |     |     |     | tationsofneuralnetworks. |     | InInternationalConference |     |
| --------- | --- | --- | --- | ------------------------ | --- | ------------------------- | --- |
onLearningRepresentations(ICLR),2024.
| Corneanu,C.A.,Escalera,S.,andMartinez,A.M. |     |     | Com- |     |     |     |     |
| ------------------------------------------ | --- | --- | ---- | --- | --- | --- | --- |
putingtheTestingErrorWithoutaTestingSet. In2020 Kornblith,S.,Norouzi,M.,Lee,H.,andHinton,G. Similar-
IEEE/CVFConferenceonComputerVisionandPattern ityofNeuralNetworkRepresentationsRevisited. May
| Recognition(CVPR).IEEE,June2020. |     |     |     | 2019. |     |     |     |
| -------------------------------- | --- | --- | --- | ----- | --- | --- | --- |
10

TowardsScalableandVersatileWeightSpaceLearning
Leclerc,G.,Ilyas,A.,Engstrom,L.,Park,S.M.,Salman, Bai,J.,andChintala,S. PyTorch: AnImperativeStyle,
H.,andMadry,A. FFCV:AcceleratingTrainingbyRe- High-PerformanceDeepLearningLibrary. InAdvances
movingDataBottleneck. InComputerVisionandPat- inNeuralInformationProcessingSystems32,2019.
ternRecognition(CVPR),2023.
Peebles,W.,Radosavovic,I.,Brooks,T.,Efros,A.A.,and
Liaw, R., Liang, E., Nishihara, R., Moritz, P., Gonzalez, Malik,J. LearningtoLearnwithGenerativeModelsof
J. E., and Stoica, I. Tune: A Research Platform for NeuralNetworkCheckpoints,September2022.
| DistributedModelSelectionandTraining. |     |     | July2018. |                        |     |                           |     |     |
| ------------------------------------- | --- | --- | --------- | ---------------------- | --- | ------------------------- | --- | --- |
|                                       |     |     |           | Ratzlaff,N.andFuxin,L. |     | HyperGAN:AGenerativeModel |     |     |
Lucas, J. R., Bae, J., Zhang, M. R., Fort, S., Zemel, R., forDiverse, PerformantNeuralNetworks. InProceed-
andGrosse,R.B. OnMonotonicLinearInterpolationof ings of the 36th International Conference on Machine
NeuralNetworkParameters. InInternationalConference Learning.PMLR,May2019.
onMachineLearning.PMLR,July2021.
|     |     |     |     | Schu¨rholt, | K., Kostadinov, | D., and | Borth, D. | Self- |
| --- | --- | --- | --- | ----------- | --------------- | ------- | --------- | ----- |
Martin,C.H.andMahoney,M.W. Rethinkinggeneraliza- Supervised Representation Learning on Neural Net-
|     |     |     |     | work Weights | for Model | Characteristic | Prediction. | In  |
| --- | --- | --- | --- | ------------ | --------- | -------------- | ----------- | --- |
tionrequiresrevisitingoldideas:Statisticalmechanicsap-
proachesandcomplexlearningbehavior,February2019a. ConferenceonNeuralInformationProcessingSystems
(NeurIPS),volume35,2021.
| Martin,C.H.andMahoney,M.W. |     | TraditionalandHeavy- |     |     |     |     |     |     |
| -------------------------- | --- | -------------------- | --- | --- | --- | --- | --- | --- |
Tailed Self Regularization in Neural Network Models. Schu¨rholt, K., Knyazev, B., Giro´-i-Nieto, X., and Borth,
January2019b. D. Hyper-RepresentationsasGenerativeModels: Sam-
|     |     |     |     | plingUnseenNeuralNetworkWeights. |     |     | InThirty-Sixth |     |
| --- | --- | --- | --- | -------------------------------- | --- | --- | -------------- | --- |
Martin,C.H.andMahoney,M.W. Heavy-tailedUniver- ConferenceonNeuralInformationProcessingSystems
salitypredictstrendsintestaccuraciesforverylargepre-
(NeurIPS),September2022a.
| traineddeepneuralnetworks. |     | InProceedingsofthe20th |     |     |     |     |     |     |
| -------------------------- | --- | ---------------------- | --- | --- | --- | --- | --- | --- |
SIAMInternationalConferenceonDataMining,2020. Schu¨rholt, K., Knyazev, B., Giro´-i-Nieto, X., and Borth,
D. Hyper-RepresentationsforPre-TrainingandTransfer
Martin, C. H. and Mahoney, M. W. Implicit self- Learning.InFirstWorkshopofPre-training:Perspectives,
| regularizationindeepneuralnetworks: |     | Evidencefrom |     |     |     |     |     |     |
| ----------------------------------- | --- | ------------ | --- | --- | --- | --- | --- | --- |
Pitfalls,andPathsForwardatICML2022,2022b.
| randommatrixtheoryandimplicationsforlearning. |     |     | The |     |     |     |     |     |
| --------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
JournalofMachineLearningResearch,22(1),January Schu¨rholt,K.,Taskiran,D.,Knyazev,B.,Giro´-i-Nieto,X.,
|     |     |     |     | andBorth,D. | ModelZoos: | ADatasetofDiversePop- |     |     |
| --- | --- | --- | --- | ----------- | ---------- | --------------------- | --- | --- |
2021.
|     |     |     |     | ulations | of Neural Network | Models. | In Thirty-Sixth |     |
| --- | --- | --- | --- | -------- | ----------------- | ------- | --------------- | --- |
Martin,C.H.,Peng,T.S.,andMahoney,M.W. Predicting ConferenceonNeuralInformationProcessingSystems
trendsinthequalityofstate-of-the-artneuralnetworks (NeurIPS)DatasetsandBenchmarksTrack,September
| withoutaccesstotrainingortestingdata. |     | NatureCommu- |     |     |     |     |     |     |
| ------------------------------------- | --- | ------------ | --- | --- | --- | --- | --- | --- |
2022c.
nications,12(1),July2021.
|                        |                              |     |     | Se´mery,O. | Osmr/imgclsmob,January2024. |     |     |     |
| ---------------------- | ---------------------------- | --- | --- | ---------- | --------------------------- | --- | --- | --- |
| Meller,D.andBerkouk,N. | SingularValueRepresentation: |     |     |            |                             |     |     |     |
ANewGraphPerspectiveOnNeuralNetworks,February Smith,L.N.andTopin,N. Super-Convergence: VeryFast
TrainingofNeuralNetworksUsingLargeLearningRates,
2023.
May2018.
Nava,E.,Kobayashi,S.,Yin,Y.,Katzschmann,R.K.,and
Grewe,B.F. Meta-LearningviaClassifier(-free)Diffu- Unterthiner,T.,Keysers,D.,Gelly,S.,Bousquet,O.,and
|               |              |     |     | Tolstikhin,I. | PredictingNeuralNetworkAccuracyfrom |     |     |     |
| ------------- | ------------ | --- | --- | ------------- | ----------------------------------- | --- | --- | --- |
| sionGuidance. | October2022. |     |     |               |                                     |     |     |     |
|               |              |     |     | Weights.      | February2020.                       |     |     |     |
Nguyen,P.,Tran,T.,Gupta,S.,Rana,S.,andDam,H.-C.
|     |     |     |     | Vaswani, | A., Ramachandran, | P., Srinivas, | A., Parmar, |     |
| --- | --- | --- | --- | -------- | ----------------- | ------------- | ----------- | --- |
HyperVAE:AMinimumDescriptionLengthVariational
Hyper-EncodingNetwork. 2019. N., Hechtman, B., and Shlens, J. Scaling Local Self-
|     |     |     |     | AttentionforParameterEfficientVisualBackbones. |     |     |     | In  |
| --- | --- | --- | --- | ---------------------------------------------- | --- | --- | --- | --- |
Nguyen,Q.N. OnConnectedSublevelSetsinDeepLearn- ProceedingsoftheIEEE/CVFConferenceonComputer
ing. InInternationalConferenceonMachineLearning, VisionandPatternRecognition,2021.
January2019.
Wang,J.,Chen,Y.,Yu,S.X.,Cheung,B.,andLeCun,Y.
Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., CompactandOptimalDeepLearningwithRecurrentPa-
Chanan,G.,Killeen,T.,Lin,Z.,Gimelshein,N.,Antiga, rameterGenerators. In2023IEEE/CVFWinterConfer-
L.,Desmaison,A.,Kopf,A.,Yang,E.,DeVito,Z.,Raison, enceonApplicationsofComputerVision(WACV).IEEE,
| M., Tejani, | A., Chilamkurthy, | S., Steiner, | B., Fang, L., | January2023. |     |     |     |     |
| ----------- | ----------------- | ------------ | ------------- | ------------ | --- | --- | --- | --- |
11

TowardsScalableandVersatileWeightSpaceLearning
Wortsman,M.,Horton,M.C.,Guestrin,C.,Farhadi,A.,and
| Rastegari,M. | LearningNeuralNetworkSubspaces. |     | In  |
| ------------ | ------------------------------- | --- | --- |
InternationalConferenceonMachineLearning.PMLR,
July2021.
| Yak,S.,Gonzalvo,J.,andMazzawi,H. |     | TowardsTaskand |     |
| -------------------------------- | --- | -------------- | --- |
Architecture-IndependentGeneralizationGapPredictors.
June2019.
Yang,Y.,Theisen,R.,Hodgkinson,L.,Gonzalez,J.E.,Ram-
| chandran,K.,Martin,C.H.,andMahoney,M.W. |     |     | Evalu- |
| --------------------------------------- | --- | --- | ------ |
atingnaturallanguageprocessingmodelswithgeneraliza-
tionmetricsthatdonotneedaccesstoanytrainingortest-
| ingdata. | TechnicalReportPreprint: | arXiv:2202.02842, |     |
| -------- | ------------------------ | ----------------- | --- |
2022.
| Zhang,C.,Ren,M.,andUrtasun,R. |     | GraphHyperNetworks     |     |
| ----------------------------- | --- | ---------------------- | --- |
| forNeuralArchitectureSearch.  |     | InInternationalConfer- |     |
enceonLearningRepresentations(ICLR),2019.
| Zhang, D.                 | W., Kofinas, | M., Zhang, Y., Chen, | Y., Burgh- |
| ------------------------- | ------------ | -------------------- | ---------- |
| outs,G.J.,andSnoek,C.G.M. |              | NeuralNetworksAre    |            |
Graphs!GraphNeuralNetworksforEquivariantProcess-
| ingofNeuralNetworks.                     |     | July2023. |        |
| ---------------------------------------- | --- | --------- | ------ |
| Zhmoginov,A.,Sandler,M.,andVladymyrov,M. |     |           | Hyper- |
Transformer:ModelGenerationforSupervisedandSemi-
| SupervisedFew-ShotLearning. |     | InInternationalConfer- |     |
| --------------------------- | --- | ---------------------- | --- |
enceonMachineLearning(ICML),January2022.
12

TowardsScalableandVersatileWeightSpaceLearning
A.AblationStudies
Inthissection,weperformablationstudiestoassesstheeffectivenessofthemethodsproposedabove: modelalignmentto
simplifythelearningtask;inferencewindowsizetoimproveinferencequality;haloingandbatch-normconditioningto
increasesamplequality.
Impact of Model Alignment. Model alignment intuitively reduces Table6. Impactofalignmentablationandpermuta-
trainingcomplexitybymappingallmodelstothesamesubspace. To tiononreconstructionloss.
|     |     |     |     |     |     | SamplePermutations |     | L   |
| --- | --- | --- | --- | --- | --- | ------------------ | --- | --- |
evaluateitsimpact,weconducttrainingexperimentswiththesamecon- rec
figurationondatasetswithandwithoutalignedmodels. Inthedataset Aligned View1 View2 Train Test
withalignedmodels,weuseeitherthealignedformor5randompermu-
|     |     |     |     |     |     | No Perm. | Perm. 0.304 | 0.167 |
| --- | --- | --- | --- | --- | --- | -------- | ----------- | ----- |
tationsforthetwoviewsforbothreconstructionandcontrastivelearn- Yes Perm. Perm. 0.148 0.082
ing. AsshowninTable6,theresultsshowtwoeffects. First,alignment Yes Align Perm. 0.107 0.082
|     |     |     |     |     |     | Yes Align | Align 0.072 | 0.082 |
| --- | --- | --- | --- | --- | --- | --------- | ----------- | ----- |
throughgitre-basinsimplifiesthelearningtaskandcontributestoim-
provedgeneralization,bothtrainingandtestlossesarereducedbymorethan50%. Second,anchoringatleastoneofthe
viewstothealignedformdoesfurtherreducethetrainingloss,butdoesnotimprovegeneralization.
ThesequentialdecompositionofSANEallows
WindowSizeAblation.
| onetopretrainnotonthefullmodelsequence,butonsubsequences. |            |                     |            |               | The           |     |     |     |
| --------------------------------------------------------- | ---------- | ------------------- | ---------- | ------------- | ------------- | --- | --- | --- |
| choice of                                                 | the length | of the subsequence, | the window | size,         | is a critical |     |     |     |
| parameterthatbalancescomputationalloadandcontext.         |            |                     |            | Weusedawindow |               |     |     |     |
of256forpretrainingformostofourexperiments.
| Here, we                                            | study the influence                               | of the          | window size on             | reconstruction   | error,   |     |     |     |
| --------------------------------------------------- | ------------------------------------------------- | --------------- | -------------------------- | ---------------- | -------- | --- | --- | --- |
| exploringvaluesrangingfrom32to2048.                 |                                                   |                 | Ourexperimentsdidnotreveal |                  |          |     |     |     |
| substantial                                         | impact of                                         | smaller windows | on pretraining             | loss or          | sampling |     |     |     |
| performance.                                        | Thisseemstosuggestthatawindowsizeaslargeas2048may |                 |                            |                  |          |     |     |     |
| stillbeinsufficientonResNetstocaptureenoughcontext. |                                                   |                 |                            | Alternatively,it |          |     |     |     |
maysuggestthattheunderlyingassumptionthatcontextmattersmaynot
entirelyholdup.
Figure6.SANEreconstructionlossovernumber
However,wedidobserveanimportantimpactontherelationshipbetween
|          |               |               |                   |        |         | oftokenswithinawindow. | Thelossislowest |     |
| -------- | ------------- | ------------- | ----------------- | ------ | ------- | ---------------------- | --------------- | --- |
| training | and inference | window sizes. | During inference, | memory | load is |                        |                 |     |
aroundthetrainingwindowsizeof256tokens,
| significantly | lower. | Inference allows | much larger | window sizes, | up to |     |     |     |
| ------------- | ------ | ---------------- | ----------- | ------------- | ----- | --- | --- | --- |
longersequencesuptothefullmodelsequence
the entire length of the ResNet sequence. However, departing from the lengthof50ktokenscauseinterferenceanddou-
trainingwindowsizeappearstointroduceinterference,whichaffectsthe
blethereconstructionerror.
reconstructionerror(Figure6).
Table7. Ablationofbatch-normconditioning
| Haloandbatch-normconditioning. |     |     | Haloingandbatch-normcondition- |     |     |     |     |     |
| ------------------------------ | --- | --- | ------------------------------ | --- | --- | --- | --- | --- |
andhaloing.
| ingaimatreducingnoiseinmodelsampling;seeSection2. |     |     |     | Toassesstheir |     |            |          |     |
| ------------------------------------------------- | --- | --- | --- | ------------- | --- | ---------- | -------- | --- |
|                                                   |     |     |     |               |     | Ep. Method | CIFAR-10 |     |
impactonsamplingperformance,weconductanin-domainexperimentus-
ingSANEtrainedonCIFAR-10ResNet-18s,usingpromptexamplesfrom randinit ∼10/%
|     |     |     | Wecomparewithna¨ıvesam- |     |     | na¨ıve | 10±0.0 |     |
| --- | --- | --- | ----------------------- | --- | --- | ------ | ------ | --- |
thetrainsetandfine-tuningonCIFAR-10.
0
plingwithouthaloingandbatch-normconditioning. TheresultsinTable Haloed 14.5±6.3
7showthesignificantimprovementsachievedbybothhaloingandbatch- BN-cond 60.8±2.2
|                   |                                              |     |     |     |     | Haloed+BN-cond | 64.8±2.1 |     |
| ----------------- | -------------------------------------------- | --- | --- | --- | --- | -------------- | -------- | --- |
| normconditioning. | Fromrandomguessingofna¨ıvesampling,combining |     |     |     |     |                |          |     |
64.4±2.9
bothimprovestoaround65%. Sincebothmethodsaimatreducingnoise randinit
|     |     |     |     |     |     | na¨ıve | 90.8±0.2 |     |
| --- | --- | --- | --- | --- | --- | ------ | -------- | --- |
forzero-shotsampling,theireffectislargestthenanddiminishessomewhat
|                   |                                               |                |               |               |      | 5 Haloed       | 90.9±0.1 |     |
| ----------------- | --------------------------------------------- | -------------- | ------------- | ------------- | ---- | -------------- | -------- | --- |
| duringfinetuning. | Bothmethodsnotonlyimprovezero-shotsamplingper |                |               |               |      |                |          |     |
|                   |                                               |                |               |               |      | BN-cond        | 90.7±0.2 |     |
| se but make       | the sampled                                   | models provide | enough signal | to facilitate | sub- |                |          |     |
|                   |                                               |                |               |               |      | Haloed+BN-cond | 90.9±0.2 |     |
samplingorbootstrappingstrategies.
13

TowardsScalableandVersatileWeightSpaceLearning
| B.SANEArchitectureDetails |     |                 | Table8. ArchitectureDetailsforSANE |      |           |
| ------------------------- | --- | --------------- | ---------------------------------- | ---- | --------- |
|                           |     | Hyper-Parameter |                                    | CNNs | ResNet-18 |
InTable8,weprovideadditionalinformationonthetraining tokensize 289 288
hyper-parametersforSANEonpopulationsofsmallCNNsas sequencelenght ∼50 ∼50k
wellasResNet18s. Thesevaluesarethestablemeanacrossall windowsize 32 256,512
|     |     | d model |     | 1024 | 2048 |
| --- | --- | ------- | --- | ---- | ---- |
experiments,exactvaluescanvaryfrompopulationtopopula-
|     |     | latent | dim | 128 | 128 |
| --- | --- | ------ | --- | --- | --- |
tion.Fullexperimentconfigurationsaredocumentedinthecode.
|     |     | transformerlayers |     | 4   | 8   |
| --- | --- | ----------------- | --- | --- | --- |
|     |     | transformerheads  |     | 4,8 | 4,8 |
C.SANEEmbeddingAnalysis-AdditionalResults
ThissectioncontainsadditionalresultsonSANEembeddinganalysis,incomparisonwithpreviousweightmatrixanalysis.
InFigure7,wecomparetheeigenvaluedistributionfordifferentmodelswithSANEembeddings. Replicatingtheexperiment
setupfrom(Martin&Mahoney,2019b;2021),wetrainMiniAlexNetmodelsonCIFAR-10varyingonlythebatchsize.With
asmallerbatchsizeandlongertrainingduration,theeigenvaluedistributiontransitionsfromrandomwithveryfewspikes,
TheembeddingsofSANEappeartoalsobecomemoreheavy-tailed,butdo
overabulkwithmanyspikes,toheavy-tailed.
notseemtopickuponthechangefromfewtomanyspikes. Theresultsaresuggestive,pointingtoobviousfollow-upwork.
Figure7.ComparisonbetweenWeightWatcherfeatures(top)andSANE(bottom).Martin&Mahoney(2019b)identifydifferentphasesin
theeigenvaluespectrumoftrainedweightmatrices.WereplicatetheexperimentsetupandfindESDssimilartorandom(topleft),bulk
andspikes(topmiddle)andheavy-tailed(topright).WecomparetheseagainstpairwisedistancesofSANEembeddingsofthesamelayer.
Whilethedistributionshaveadifferentshape,itappearstobecomemoreheavy-tailedgoingfromrandomtoheavytailed.
Figures8and9compareSANEwithdifferentWeightWatchermetricsonVGGsfrompytorchcv(Se´mery,2024)andthe
ResNet-18zoofromthemodelzoodataset(Schu¨rholtetal.,2022c).
D.ModelPropertyPrediction-AdditionalResults
Inthissection,weprovideadditionaldetailsforSection5.1. Table9showsfullresultsforpopulationsofsmallCNNs.
Table9. PropertypredictiononpopulationsofsmallCNNs.
| MNIST       | SVHN        | CIFAR-10(CNN) |      | STL    |      |
| ----------- | ----------- | ------------- | ---- | ------ | ---- |
| W s(W) SANE | W s(W) SANE | W s(W)        | SANE | W s(W) | SANE |
ACC 0.965 0.987 0.978 0.910 0.985 0.991 -7.580 0.965 0.885 -18.818 0.919 0.305
Epoch 0.953 0.974 0.958 0.833 0.953 0.930 0.636 0.923 0.771 -1.926 0.977 0.344
Ggap 0.246 0.393 0.402 0.479 0.711 0.760 0.324 0.909 0.772 -0.617 0.858 0.307
14

TowardsScalableandVersatileWeightSpaceLearning
Figure8.ComparisonbetweendifferentWeightWatcher(WW)features(left)andSANE(right).FeaturesoverlayerindexforVGGsfrom
pytorchcvofdifferentsizes.SANEshowssimilartrendstoWW,lowvaluesatearlylayersandasharpincreaseattheend.
Figure9.ComparisonbetweendifferentWeightWatcher(WW)features(left)andSANE(right).FeaturesoverlayerindexforResnets
frompytorchcvofdifferentsizes.SANEshowssimilartrendstoWW,lowvaluesatearlylayersandasharpincreaseattheend.
15

TowardsScalableandVersatileWeightSpaceLearning
Figure10.ComparisonbetweenWeightWatcherfeatures(left)andSANE(right).AccuracyovermodelfeaturesforResnetsandVGGs
frompytorchcvofdifferentsizes.SANEshowssimilartrendstoWW,lowvaluesatearlylayersandasharpincreaseattheend.
Figure11.ComparisonbetweenWeightWatcherfeatures(left)andSANE(right). AccuracyovermodelfeaturesforResNetsfromthe
ResNetmodelzoo. AlthoughSANEispretrainedinaself-supervisedfashion,itpreservesthelinearrelationofaglobally-aggregated
embeddingtomodelaccuracy.
D.1.ComparisontoPreviousWork
Here,wecompareSANEwithpreviousworktodisseminatetheinformationcontainedinmodelembeddings.Theexperiment
setupinthispaperisdesignedaroundtheResNets,andthereforeitusessparseepochsforcomputationalefficiency. For
consistency,weusethesamesetupfortheCNNzoosaswell. Theexactnumbersarethereforenotdirectlycomparableto
Schu¨rholtetal.(2021). Toprovideasmuchcontextaspossible,weapproachthecomparisonfromtwoangles:
16

TowardsScalableandVersatileWeightSpaceLearning
(1) Directcomparisontothepublishedresults: tocontextualize,weusethe(deterministic)resultsofweightstatistics
s(W)toadjustforthedifferencesinsetup. Wemarktheresultsfors(W)fromSchu¨rholtetal.(2021)ass(W) and
pp
comparetotheirE c+ Dwherepossible.
(2) Approximation of the effect of global embeddings: previous work used global model embeddings, which we
approximate by using the full model embedding sequence. We therefore compare SANE+ aggregated tokens (as
proposedinthesubmission)toSANE+fullmodelsequence(similartoSchu¨rholtetal.(2021)).
TheresultsintheTablesbelowallowthefollowingconclusions:
(1) SANEmatchestheperformanceofpreviouswork: TheonlydataavailablefordirectcomparisonistheMNIST
zoo. Here, both in direct comparison and in relation to s(W) cross-relating our results with published numbers,
SANEmatchespublishedperformanceofE +D. Onotherzoos, E +D hadsimilarperformancetos(W). We
c c
likewisefindSANEembeddingstohavesimilarperformancetos(W)inourexperiments.
(2) SANE+ full sequence improves downstream task performance over the SANE+ aggregated sequence: That
indicates that SANE+ full sequence contains more information for model prediction. However, both Schu¨rholt
etal.(2021)andSANEwithfullsequencehavethedisadvantagethattheydonotscale. Withgrowingmodels,the
representation learner of Schu¨rholt et al. (2021) and the input to the linear probe of SANE+ full sequence grow
accordingly. SANE+aggregatedsequencedoeslosesomeinformationonsmallmodels,butscalesgracefullytolarge
modelsandremainscompetitive.
Table10.PropertyPredictioncomparisontopreviousworkontheMNIST-CNNmodelzoo.Wecompareourlinearprobingresultsfrom
weightsW,layer-wisequintiless(W),embeddingsfromSANEeitheraggregatedintooneembeddingorusingthefullsequencetoresults
previouslypublishedinSchu¨rholtetal.(2021).Wemarktheirresultsfors(W)ass(W) .Sincetheexperimentalsetupisnotthesame,
pp
thenumbersofs(W)donotmatch.
| W           | s(W) SANEaggregated |       | SANEfullsequence | s(W)  | E D   |
| ----------- | ------------------- | ----- | ---------------- | ----- | ----- |
|             |                     |       |                  | pp    | c+    |
| ACC 0.965   | 0.987               | 0.978 | 0.987            | 0.977 | 0.973 |
| Epoch 0.953 | 0.974               | 0.958 | 0.975            | 0.987 | 0.989 |
| Ggap 0.246  | 0.393               | 0.402 | 0.461            | 0.662 | 0.667 |
Table11.PropertyPredictioncomparisontopreviousworkontheSVHN-CNNmodelzoo.Wecompareourlinearprobingresultsfrom
weightsW,layer-wisequintiless(W),toembeddingsfromSANEeitheraggregatedintooneembeddingorusingthefullsequence.For
thiszoo,previousresultsarenotavailable.
| W           | s(W) SANEaggregated |       | SANEfullsequence | s(W) | E D |
| ----------- | ------------------- | ----- | ---------------- | ---- | --- |
|             |                     |       |                  | pp   | c+  |
| ACC 0.910   | 0.985               | 0.991 | 0.993            | n/a  | n/a |
| Epoch 0.833 | 0.953               | 0.930 | 0.943            | n/a  | n/a |
| Ggap 0.479  | 0.711               | 0.760 | 0.77             | n/a  | n/a |
Table12.PropertyPredictioncomparisontopreviousworkontheCIFAR-CNN(m)modelzoo.Wecompareourlinearprobingresults
fromweightsW,layer-wisequintiless(W),toembeddingsfromSANEeitheraggregatedintooneembeddingorusingthefullsequence.
Forthiszoo,previousresultsarenotavailable.
| W           | s(W)  | SANEaggregated | SANEfullsequence | s(W) | E D |
| ----------- | ----- | -------------- | ---------------- | ---- | --- |
|             |       |                |                  | pp   | c+  |
| ACC -7.580  | 0.965 | 0.885          | 0.947            | n/a  | n/a |
| Epoch 0.636 | 0.923 | 0.771          | 0.879            | n/a  | n/a |
| Ggap 0.324  | 0.909 | 0.772          | 0.811            | n/a  | n/a |
E.ModelGeneration-AdditionalResults
Thissectioncontainsadditionalresultsfrommodelsamplingexperiments,extendingSection5.2. InTable13,weshow
resultsonsmallCNNstransferringtoanewtask. Similarly,Table14showsresultsonResNet-18modelsfortasktransfers.
17

TowardsScalableandVersatileWeightSpaceLearning
Lastly,Tables15,16and17containadditionalresultsfortransferringfromResNet-18CIFAR-100toResNet34and/or
Tiny-Imagenet.
Table13.ModelgenerationonCNNmodelpopulationstransferlearnedonanewtask.Wecomparesampledmodelsatdifferentepochs
withmodelstrainedfromscratchandmodelsfine-tunedfromtheanchorsamples.
Method SVHNtoMNIST CIFAR-10toSTL-10
Epoch0 Epoch1 Epoch25 Epoch0 Epoch1 Epoch25
tr.fr.scratch ∼10/% 20.6+-1.6 83.3+-2.6 ∼10/% 21.3+-1.6 44.0+-1.0
pretrained 29.1+-7.2 84.1+-2.6 94.2+-0.7 16.2+-2.3 24.8+-0.8 49.0+-0.9
S 31.8+-5.6 86.9+-1.4 95.5+-0.4 n/a n/a n/a
KDE30
SANE 40.2+-4.8 86.7+-1.6 94.8+-0.4 15.5+-2.3 24.9+-1.6 49.2+-0.5
KDE30
SANE . 37.9+-2.8 88.2+-0.5 95.6+-0.3 17.4+-1.4 25.6+-1.7 49.8+-0.6
SUB
E.1.Diversityofsampledmodels
AninterestingquestioniswhethersamplingSANEgeneratesversionsofthesamemodel. Totestthat, weevaluatethe
diversityofsamplesgeneratedwithonlyafewfew-shotexamplesbycombiningthemodelstoensembles.Theimprovements
oftheensemblesovertheindividualmodelsdemonstratetheirdiversity.Thisindicatesthatgivenveryfew,early-stageprompt
examples,samplinghyper-representationsimproveslearningspeedandperformanceinotherwiseequalsettings.Additionally,
weconductedexperimentswithvaryingnumbersofpromptexamples, revealingthatincreasingthenumberofprompt
examplesenhancesbothperformanceanddiversity. Nonetheless,evenasinglepromptexampletrainedforjust2epochs
containssufficientinformationtogeneratemodelsamplesthatsurpassthosederivedfromrandominitialization;seeTable17.
18

TowardsScalableandVersatileWeightSpaceLearning
Table14.ModelgenerationonResNet-18modelpopulationstransferredtoanewtask.Wecomparesampledmodelsatdifferenttransfer
learningepochswithmodelstrainedfromscratchandmodelsfine-tunedfromthesameanchorsamples.
Epoch Method CIFAR-10toCIFAR-100 CIFAR-100toTiny-Imagenet Tiny-ImagenettoCIFAR-100
| tr.fr.scratch | ∼1/%      | ∼0.5/%    |     | ∼1/%       |
| ------------- | --------- | --------- | --- | ---------- |
| Finetuned     | 1.0+-0.3  | 0.5+-0.0  |     | 1.1+-0.2   |
| 0 SANEKDE30   | 1.0+-0.3  | 0.5+-0.1  |     | 1.0+-0.2   |
| SANESUB       | 1.0+-0.3  | 0.6+-0.0  |     | 1.1+-0.2   |
| SANEBOOT      | 1.1+-0.2  | 0.5+-0.0  |     | 0.9+-0.2   |
| tr.fr.scratch | 17.5+-0.7 | 13.8+-0.8 |     | 17.5+-0.7  |
| Finetuned     | 27.5+-1.3 | 25.7+-0.5 |     | 51.7+-0.5  |
| 1 SANEKDE30   | 26.8+-1.4 | 21.5+-0.9 |     | 40.2+-1.0  |
| SANESUB       | 26.4+-1.9 | 21.5+-1.0 |     | 40.63+-1.3 |
| SANEBOOT      | 25.701.9  | 21.7+-1.0 |     | 40.9+-0.8  |
| tr.fr.scratch | 36.5+-2.0 | 31.1+-1.6 |     | 36.5+-2.0  |
| Finetuned     | 45.7+-1.0 | 36.3+-2.5 |     | 52.6+-1.3  |
| 5 SANEKDE30   | 44.5+-2.0 | 36.3+-1.2 |     | 47.2+-3.3  |
| SANESUB       | 45.6+-1.2 | 35.8+-1.4 |     | 49.8+-2.3  |
| SANEBOOT      | 43.3+-2.4 | 37.3+2.0  |     | 50.2+-3.4  |
| tr.fr.scratch | 53.3+-2.0 | 38.5+-1.9 |     | 53.3+-2.0  |
| Finetuned     | 71.9+-0.1 | 63.4+-0.2 |     | 73.9+-0.3  |
| 15 SANEKDE30  | 71.8+-0.3 | 63.6+-0.2 |     | 73.4+-0.2  |
SANESUB
|                  | 72.0+-0.2 | 63.6+-0.3 |     | 73.5+-0.2 |
| ---------------- | --------- | --------- | --- | --------- |
| SANEBOOT         | 71.9+-0.3 | 63.4+-0.1 |     | 73.7+-0.3 |
| 25 tr.fr.scratch | 56.5+-2.0 | 43.3+-1.9 |     | 56.5+-2.0 |
| 50 tr.fr.scratch | 70.7+-0.4 | 57.3+-0.6 |     | 70.7+-0.4 |
| 60 tr.fr.scratch | 74.2+-0.3 | 63.9+-0.5 |     | 74.2+-0.3 |
Table15. Few-shotmodelgenerationforanewtask:SamplingResNet-34modelsforCIFAR-100.SANEwaspretrainedonCIFAR-100
ResNet-18s,5samplesaredrawnusingsubsampling.Togetpromptexamples,wetrain3ResNet-34modelsonCIFAR-100for2epochs
toameanaccuracyof26%.
CIFAR100ResNet-18toResNet-34
|     | Ep. Method      | 5Epochs  | 15Epochs |     |
| --- | --------------- | -------- | -------- | --- |
|     | 0 tr.fr.Scratch | 1.0±0.1  | 1.0±0.1  |     |
|     | SANE            | 1.6±0.3  | 1.6±0.3  |     |
|     | 1 tr.fr.Scratch | 12.4±1.0 | 12.9±0.8 |     |
|     | SANE            | 16.8±0.7 | 23.1±0.3 |     |
|     | 5 tr.fr.Scratch | 49.5±0.6 | 36.2±1.7 |     |
SANE
|     |                  | 51.9±0.6 | 37.8±1.4 |     |
| --- | ---------------- | -------- | -------- | --- |
|     | 15 tr.fr.scratch |          | 68.8±0.4 |     |
SANE
69.3±0.3
|     | SANEEns. | 53.5 | 71.3 |     |
| --- | -------- | ---- | ---- | --- |
Table16. Few-shotmodelgenerationforanewtaskandarchitecture:SANEtrainedonCIFAR-100ResNet-18susedtogenerateResNet-
34sforTiny-Imagenet.5samplesaredrawnusingsubsampling.Togetpromptexamples,wetrain3ResNet-34modelsonTiny-Imagenet
for2epochstoameanaccuracyof28.5%.
ResNet-18CIFAR100toResNet-34Tiny-Imagenet
|     | Ep. Method    | 5epochs | 15epochs |     |
| --- | ------------- | ------- | -------- | --- |
|     | tr.fr.Scratch | 0.5±0.0 | 0.5±0.0  |     |
0
|     | SANE          | 0.5±0.1  | 0.6±0.2  |     |
| --- | ------------- | -------- | -------- | --- |
|     | tr.fr.Scratch | 10.5±1.4 | 11.9±1.9 |     |
1
|     | SANE          | 13.3±0.5 | 18.5±0.7 |     |
| --- | ------------- | -------- | -------- | --- |
|     | tr.fr.Scratch | 47.2±0.7 | 31.1±1.7 |     |
5 SANE
|     |               | 50.6±0.3 | 31.6±0.6 |     |
| --- | ------------- | -------- | -------- | --- |
|     | tr.fr.Scratch |          | 61.9±0.3 |     |
|     | 15 SANE       |          | 62.7±0.3 |     |
|     | SANEEns.      | 52       | 65.1     |     |
19

TowardsScalableandVersatileWeightSpaceLearning
SANEwaspretrainedonCIFAR-100ResNet-18s,5samplesaredrawnusing
Table17. SamplingResNet-34modelsforCIFAR-100.
subsampling.Togetpromptexamples,wetrainasingleResNet-34modelonCIFAR-100for2epochstoanaccuracyof26%.
CIFAR100ResNet-18toResNet-34
| Epoch    | Method        | 5Epochs   | 15Epochs  |
| -------- | ------------- | --------- | --------- |
| 0        | tr.fr.Scratch | 1.0+-0.1  | 1.0+-0.1  |
|          | SANE          | 1.5+-0.2  | 1.6+-0.1  |
| 1        | tr.fr.Scratch | 12.4+-1.0 | 12.9+-0.8 |
|          | SANE          | 16.9+-0.7 | 19.4+-0.2 |
| 5        | tr.fr.Scratch | 49.5+-0.6 | 36.2+-1.7 |
|          | SANE          | 51.5+-0.3 | 38.6+-1.6 |
| 15       | tr.fr.scratch |           | 68.8+-0.4 |
|          | SANE          |           | 69.1+-0.1 |
| Ensemble | SANE          | 51.8      | 70.2      |
20