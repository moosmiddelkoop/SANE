000
001
|     |     | Utilizing |     | Weight |     | Space | Learning |     | for | Data-Free |     | Model | Editing |     |     |
| --- | --- | --------- | --- | ------ | --- | ----- | -------- | --- | --- | --------- | --- | ----- | ------- | --- | --- |
002
003
004
005
| 006 |     |     |     |     |     |     | AnonymousAuthors1 |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | --- | --- | --- | --- | --- | --- | --- |
007
008
| 009 |     |     |     | Abstract |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
010
|     |     | The growing | availability |     | of model | checkpoints |     |     |     |     |     |     |     |     |     |
| --- | --- | ----------- | ------------ | --- | -------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
011
| 012 |     | has enabled | Weight | Space | Learning | (WSL), |     | a   |     |     |     |     |     |     |     |
| --- | --- | ----------- | ------ | ----- | -------- | ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- |
deeplearningparadigmthatusesmodelparame-
013
| 014 |     | tersasadatamodality. |     | WhileWSLhasbeenused |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | -------------------- | --- | ------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
forlearningfunctionsofneuralnetworkweights,
015
|     |     | like accuracy | prediction, |     | its potential | for | direct |     |     |     |     |     |     |     |     |
| --- | --- | ------------- | ----------- | --- | ------------- | --- | ------ | --- | --- | --- | --- | --- | --- | --- | --- |
016
| 017 |     | modeleditingremainsunderexplored. |     |     |     |     | Oneuse- |     |     |     |     |     |     |     |     |
| --- | --- | --------------------------------- | --- | --- | --- | --- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
fulcaseofmodeleditingismachineunlearning,
018
| 019 |     | where the        | reliance | on original          |     | or target | data        | to  |     |     |     |     |     |     |     |
| --- | --- | ---------------- | -------- | -------------------- | --- | --------- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- |
|     |     | forget currently |          | limits applicability |     | in        | strict pri- |     |     |     |     |     |     |     |     |
| 020 |     |                  |          |                      |     |           |             |     |     | 1.0 |     |     |     |     |     |
vacyandlegacysystems. Weillustratethepoten- Before intervention
021
After intervention
022 tialofutilizingWSLfordata-freemodelediting, Predicted After
|     |     |     |     |     |     |     |     |     |     | 0.8 |     |     | Target |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | --- | --- |
throughexamplesofzero-shotunlearningandper-
023
| 024 |     | formanceimprovement. |     |     | Wetrainametanetwork |     |     |     |     | ycaruccA 0.6 |     |     |     |     |     |
| --- | --- | -------------------- | --- | --- | ------------------- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- |
tocapturetheoptimizationlandscapeoftheSmall
025
|     |     | CNNModelZoo,predictingclassrecallwithhigh |     |     |     |     |     |     |     | 0.4 |     |     |     |     |     |
| --- | --- | ----------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
026
R2
| 027 |     | accuracy | (mean | ≈   | 0.842). | By backpropa- |     |     |     |     |     |     |     |     |     |
| --- | --- | -------- | ----- | --- | ------- | ------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
0.2
|     |     | gating a | loss through | the | metanetwork |     | with | re- |     |     |     |     |     |     |     |
| --- | --- | -------- | ------------ | --- | ----------- | --- | ---- | --- | --- | --- | --- | --- | --- | --- | --- |
028
| 029 |     | specttoitsinputweights,weobtaindirectionsin |     |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | ------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
0.0
weightspacethatselectivelyedittarget-classper- 0 1 2 3 4 5 6 7 8 9
| 030 |     |           |                                 |     |     |     |     |     |                                                       |     |     |     | Classes |     |     |
| --- | --- | --------- | ------------------------------- | --- | --- | --- | --- | --- | ----------------------------------------------------- | --- | --- | --- | ------- | --- | --- |
|     |     | formance. | Onheld-outmodels,thisachievesup |     |     |     |     |     |                                                       |     |     |     |         |     |     |
| 031 |     |           |                                 |     |     |     |     |     | Figure1.Top:AcollectionofCNNmodelsareflattenedandused |     |     |     |         |     |     |
032 to52percentrecalldropsonforgetclasseswhile totrainametanetworktopredictmultivariateclassrecall.Atask
preservingretain-classaccuracy. specificlossfunctionw.r.t.themetanetworkinputs,i.e.theCNN
033
parameters,canbeusedtoeditperformanceofsimilarbutbefore
034
unseenCNNmodels.Bottom:aneffectiveexampleresultofour
035
|     |     |     |     |     |     |     |     |     | method. | Weunlearnclass4inaFashion-MNISTclassification |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | --------------------------------------------- | --- | --- | --- | --- | --- |
036 1.Introduction CNN:SmallCNNZoomodel#86whichwasnotyetseenbythe
| 037 |     |     |     |     |     |     |     |     | metanetwork. |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- | --- |
InAIinterpretabilityresearch,neuralnetworksareusually
038
studiedthroughthelensoftheirinput-activation-outputbe-
039
040 havior. Recentworkhighlightsthatinsightsintotheirmech- space learning (WSL), which establishes neural network
anisticfunctionalitycanbegainedbyexaminingtheirpa-
041 weights itself as a new data modality, driven by the vast
rametersdirectly(Braunetal.,2025;Bushnaqetal.,2025).
| 042 |     |     |     |     |     |     |     |     | andever-growingnumberofpubliclyavailableneuralnet- |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- | --- |
Thisperspectivetreatstheweightsofatrainedmodelasa workmodels. Acollectionofneuralnetworkweightsused
043
|     | vectorinahighdimensionalspace,theweightspace. |     |     |     |     |     |     | The |                                          |     |     |     |     |     |          |
| --- | --------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | ---------------------------------------- | --- | --- | --- | --- | --- | -------- |
| 044 |                                               |     |     |     |     |     |     |     | forweightspacelearningiscalledamodelzoo. |     |     |     |     |     | Metanet- |
045 corehypothesisisthatinformationaboutwhataneuralnet- worksaretrainedonthesemodelzoostoinferwhatamodel
|     | work | ‘knows’ | or what | it is capable |     | of is encoded |     | into its |     |     |     |     |     |     |     |
| --- | ---- | ------- | ------- | ------------- | --- | ------------- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
046 has learned without requiring explicit access to the orig-
|     | parameters,andthatthisinformationcanbeextracted. |     |     |     |     |     |     | This |      |          |          |      |                        |     |       |
| --- | ------------------------------------------------ | --- | --- | --- | --- | --- | --- | ---- | ---- | -------- | -------- | ---- | ---------------------- | --- | ----- |
| 047 |                                                  |     |     |     |     |     |     |      | inal | training | methods, | data | or model architecture. |     | Using |
isinlinewiththemachinelearningsubfieldcalledweight metanetworkstostudylargecollectionsofmodelsshows
048
049 1AnonymousInstitution,AnonymousCity,AnonymousRegion, that weight configurations can reveal training data, tasks,
| 050 |                   |     |                   |     |     |                 |     |     | hyper-parameters |     |     | and performance | of a | model | (Eilertsen |
| --- | ----------------- | --- | ----------------- | --- | --- | --------------- | --- | --- | ---------------- | --- | --- | --------------- | ---- | ----- | ---------- |
|     | AnonymousCountry. |     | Correspondenceto: |     |     | AnonymousAuthor |     |     |                  |     |     |                 |      |       |            |
etal.,2020;Unterthineretal.,2021).
051 <anon.email@domain.com>.
052
|     |     |     |     |     |     |     |     |     | Simultaneously, |     | the | need | for model editing | is  | becoming |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- | --- | ---- | ----------------- | --- | -------- |
Preliminarywork.UnderreviewbytheInternationalConference
| 053 |     |     |     |     |     |     |     |     | morerelevant. |     | After(pre-)training,somefunctionalityof |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | --------------------------------------- | --- | --- | --- | --- |
onMachineLearning(ICML).Donotdistribute.
054
1

UtilizingWeightSpaceLearningforData-FreeModelEditing
055 a model might need change. Improving performance on structureabouttheoptimizationlandscapetoprovideanac-
056 somedownstreamtask,removingbiasorspecifictraining tionableeditingsignal,thusestablishingaproofofconcept
057 data. Partlyduetolegalrequirementssuchastherightto forweight-spacemodelediting.
058 erasureestablishedintheEU’sGeneralDataProtectionand
Ourcontributionsareasfollows:
059 Regulation,orethicaldemandssuchasmodelalignment,we
060 mightaimtoremovetheinfluenceofspecificdataortasks
061 fromatrainedmodel. Astraightforwardoptionistoeditthe 1. Weshowametanetworkcanbetrainedtopredictthe
per-classrecallwithhighperformance.
062 datasetandretrainthemodel,butthescaleofmanycurrent
| 063 | state-of-the-artmodelsinusemakesthisimpractical. |     |     |     |     |     | This |     |     |     |     |     |     |     |
| --- | ------------------------------------------------ | --- | --- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
064 needhasdrivenmachineunlearning: selectivelyremoving 2. Wedevelopadata-freemodeleditingalgorithmthat
usesgradientsbackpropagatedthroughthismetanet-
| 065 | learnedinformationwithoutfullretraining. |     |     |     |     | However,most |     |     |     |     |     |     |     |     |
| --- | ---------------------------------------- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- | --- |
066 methodsassumeaccesstotrainingorforgetdataimpractical worktoadjusttheweightsofunseenneuralnetworks.
067
whensuchdataisproprietaryorrestricted.
| 068 |                        |               |     |                              |         |             |           | 3. Wedemonstrateeffectivenessontwoeditingtasks:   |     |     |     |     |     | tar- |
| --- | ---------------------- | ------------- | --- | ---------------------------- | ------- | ----------- | --------- | ------------------------------------------------- | --- | --- | --- | --- | --- | ---- |
|     | Weight                 | interpolation |     | shows                        | that to | some extent | these ed- |                                                   |     |     |     |     |     |      |
| 069 |                        |               |     |                              |         |             |           | getedforgettingandperformanceimprovement,show-    |     |     |     |     |     |      |
|     | its                    | can be made   | in  | the parameters               |         | of a model  | directly  |                                                   |     |     |     |     |     |      |
| 070 |                        |               |     |                              |         |             |           | ingselectivemodificationofclass-specificbehavior. |     |     |     |     |     |      |
|     | (Ainsworthetal.,2022). |               |     | Ilharcoetal.(2022b)showthata |         |             |           |                                                   |     |     |     |     |     |      |
071
taskorconceptcanbeisolatedbythedirectioninweight
| 072 |     |     |     |     |     |     |     | 4. WeextendtheSmallCNNModelZoo,bycomputing |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------ | --- | --- | --- | --- | --- | --- |
spaceamodelmoveswhenfinetuningonthatspecifictask.
| 073 |                                         |              |     |        |       |                   |           | per-classaccuracy(classrecall)andmakethisdataset |     |     |     |     |     |     |
| --- | --------------------------------------- | ------------ | --- | ------ | ----- | ----------------- | --------- | ------------------------------------------------ | --- | --- | --- | --- | --- | --- |
|     | Thus,                                   | task vectors | in  | weight | space | can be identified | that      |                                                  |     |     |     |     |     |     |
| 074 |                                         |              |     |        |       |                   |           | extensionpubliclyavailable.                      |     |     |     |     |     |     |
|     | pointtowardsimprovementonaspecifictask. |              |     |        |       |                   | Thesetask |                                                  |     |     |     |     |     |     |
075
vectorscanalsobeusedtoperformtaskarithmeticinweight
076
space,suchasnegationforunlearning,orrecombinationfor Thisworkis,tothebestofourknowledge,thefirstdemon-
077
strationofaddingalossfunctiontotheoutputofametanet-
transferlearning.
078
|     |     |     |     |     |     |     |     | work and | using | input attribution | to  | make changes |     | to the |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ----- | ----------------- | --- | ------------ | --- | ------ |
079 Weintroduceaweightspacelearningperspectiveonmodel
weightsofthatinputmodelinstance,withthegoalofedit-
080 editing. We propose to train a metanetwork that predicts ingfunctionalityinapretrainedmodel.
| 081 | performance                              |     | on specific | tasks | based | on the | weights of a |     |     |     |     |     |     |     |
| --- | ---------------------------------------- | --- | ----------- | ----- | ----- | ------ | ------------ | --- | --- | --- | --- | --- | --- | --- |
| 082 | largesetofmodelsfromthesamemodel-family. |     |             |       |       |        | Awell-       |     |     |     |     |     |     |     |
2.RelatedWork
| 083 | trained | metanetwork |     | implicitly | encodes | the | optimization |     |     |     |     |     |     |     |
| --- | ------- | ----------- | --- | ---------- | ------- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- |
084 landscape;itsinputgradientscanthusserveastaskvectors,
2.1.WeightSpaceLearning
085 pointersinweightspacefortargetedediting.
| 086 |     |     |     |     |     |     |     | TwopioneeringworksinweightspacelearningarebyUn- |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------------------------------------- | --- | --- | --- | --- | --- | --- |
Inpracticeweproposetobackpropagatethegradientofa
| 087 |     |     |     |     |     |     |     | terthineretal.(2021)andEilertsenetal.(2020). |     |     |     |     | Bothsuc- |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------------------------------- | --- | --- | --- | --- | -------- | --- |
taskspecificlossfunctionthroughtheregressorwithrespect cessfullydemonstrateusingamodelzootopredictmodel
088
|     | toitsinputs: |     | themodelweights. |     | Thisallowsfordata-free |     |     |                                                   |     |     |     |     |     |     |
| --- | ------------ | --- | ---------------- | --- | ---------------------- | --- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| 089 |              |     |                  |     |                        |     |     | propertieswithameta-networkusingmodelweightsasin- |     |     |     |     |     |     |
gradientdescentforiterativelyeditingmodelperformance
090 puts. The works differ mainly in their prediction target:
|     | onanyquantifiableanddifferentiabletask.     |     |     |     |     | Wedemonstrate |            |                                                   |     |     |     |     |     |     |
| --- | ------------------------------------------- | --- | --- | --- | --- | ------------- | ---------- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| 091 |                                             |     |     |     |     |               |            | Unterthineretal.(2021)predictmodelperformancefrom |     |     |     |     |     |     |
|     | thisthroughontwoclass-specificeditingtasks: |     |     |     |     |               | forgetting |                                                   |     |     |     |     |     |     |
092 weights, whereas Eilertsen et al. (2020) predict different
andlearning.OurmethodissummarizedinFigure1.Specif- training algorithm hyperparameters from weights. Using
093
ically,wetrainametanetworktopredictper-classaccuracy
| 094 |     |     |     |     |     |     |     | meta-classifiers,theydemonstratethatevensmallsubsets |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------------------------------------------- | --- | --- | --- | --- | --- | --- |
ofCNNmodelstheSmallCNNZoomodelcollection(Un-
| 095 |     |     |     |     |     |     |     | of weights | encode | rich information |     | about | datasets, | opti- |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ------ | ---------------- | --- | ----- | --------- | ----- |
terthineretal.,2021),amorefine-grainedtaskthanoverall
| 096 |                                        |     |             |     |      |          |              | mizers,                                            | and activations. | We  | extend | the methodology |     | of    |
| --- | -------------------------------------- | --- | ----------- | --- | ---- | -------- | ------------ | -------------------------------------------------- | ---------------- | --- | ------ | --------------- | --- | ----- |
|     | performance                            |     | prediction. | We  | show | that the | metanetwork  |                                                    |                  |     |        |                 |     |       |
| 097 |                                        |     |             |     |      |          |              | Unterthineretal.(2021)byusingametanetworktopredict |                  |     |        |                 |     |       |
|     | predictsclassrecallwithhighaccuracy(R2 |     |             |     |      |          | ≈ 0.84),dis- |                                                    |                  |     |        |                 |     |       |
|     |                                        |     |             |     |      |          |              | theper-classrecalloftheSmallCNNZoomodels.          |                  |     |        |                 |     | Tothe |
098
|     | entanglingclass-specificfunctionality. |             |           |     |             | Onheld-outmodels |              |                                                     |     |     |     |     |     |     |
| --- | -------------------------------------- | ----------- | --------- | --- | ----------- | ---------------- | ------------ | --------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| 099 |                                        |             |           |     |             |                  |              | bestofourknowledgethisisthefirstdemonstrationofsuch |     |     |     |     |     |     |
|     | never                                  | seen during | training, | the | metanetwork |                  | provides ac- |                                                     |     |     |     |     |     |     |
| 100 |                                        |             |           |     |             |                  |              | fine-grainedperformancepredictionbyametanetwork.    |     |     |     |     |     |     |
tionableeditingsignals,throughthebackpropagationofa
101
lossfunctionw.r.t. themetanetwork’sinputs: theoriginal In recent years, impressive steps have been made to ad-
102
CNNmodelweights. dress symmetries, scaling, and generalization to diverse
| 103 |     |     |     |     |     |     |     | feedforwardarchitecturesinWSL(Schu¨rholtetal.,2021; |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------------------------------------- | --- | --- | --- | --- | --- | --- |
104 Asagenericapproachthatoperatessolelyonweightvectors
2022a;b;Navonetal.,2023;Zhouetal.,2023;Limetal.,
| 105 | without | task-specific |     | design, | we do | not expect | to outper- |     |     |     |     |     |     |     |
| --- | ------- | ------------- | --- | ------- | ----- | ---------- | ---------- | --- | --- | --- | --- | --- | --- | --- |
2024;Kalogeropoulosetal.,2024;Schu¨rholtetal.,2024;
| 106 | form | dedicated | unlearning | methods |     | that leverage | model |     |     |     |     |     |     |     |
| --- | ---- | --------- | ---------- | ------- | --- | ------------- | ----- | --- | --- | --- | --- | --- | --- | --- |
Knyazevetal.,2023;Wangetal.,2025;Liangetal.,2025).
| 107 | inferenceorsyntheticdatageneration. |     |     |     |     | Rather,ouraimis |     |     |     |     |     |     |     |     |
| --- | ----------------------------------- | --- | --- | --- | --- | --------------- | --- | --- | --- | --- | --- | --- | --- | --- |
Amoredetailedoverviewofthedevelopmentsofthefield
108 todemonstratethataWSLmetanetworkencodessufficient isgiveninAppendixA.
109
2

UtilizingWeightSpaceLearningforData-FreeModelEditing
110 Concurrent to work presented here, Moulin et al. (2025) impractical1andproposezero-shotunlearning: ”requiring
111 havetakenasimilarapproachofusingatrainedmetanet- notrainingsamplesorinformationrelatedtothetraining
112 work to inform a loss function of which the gradient is process”. Somerecentworkhasrelaxedthisdefinitionto
113 backpropagated w.r.t. the input weights; albeit as an ele- permit forget-set access (Foster et al., 2024a). However,
114 mentwithinalargertrainingframework,andusedduring our approach adheres to the strict original definition by
115 optimization–whereas we aim to directly edit pre-trained Chundawatetal.(2023). Asasolutiontheyproposetwo
116 models. Specifically,theyshowthatWSLcanalsobeused methods: Error minimizing-maximizing noise and gated
117 goguidegeneralizationinRL.Theauthorstrainamodel knowledgetransfer(GKT).Bothmethodsrequirefullac-
118 zoo of RL policies on two toy problems and then train a cesstoaworkingmodel,whereasourweightspacelearning
119 metanetworktopredictthegeneralizabilityofthesepolicies. basedmethodonlyneedstheweightvector. Furthermore,
120 Thismetanetworkisusedtoinformanadditionallosscom- theirbestmethodrequirestrainingageneratorfromscratch
121 ponenttothePPOalgorithm(Schulmanetal.,2017)during per model; our method front-loads the computation into
122 trainingofnewpolicies. Thisleadstoimprovedgeneraliza- metanetwork training, then editing is just a few gradient
123 tion in the policies trained with this metanetwork-guided stepsforeverymodelyouwishtoeditafterpre-training.
124 lossfunctionversusthosetrainedwithout.
Recently, Mishra et al. (2025) have proposed data-free
125
Rangeletal.(2024)presentedaweightspacelearningper- unlearning for CLIP-style multimodal models (Radford
126
spectiveonmachineunlearning. Self-supervisedlearning et al., 2021). They exploit the joint representation space
127
(SSL) metanetworks can create hyper-representations of by computing text embeddings from class names via the
128
modelweights. Theirmethodachievesunlearningbygen- textencoder,andorthogonalizingimagefeaturesawayfrom
129
eratingunlearnedmodels,usingdiffusiontransformerhy- theseembeddings. Incontrast,ourmetanetwork-basedap-
130
pernetworks,whicharetrainedonmodelswithknowledge proachisarchitecture-agnosticinprinciple: givenasuitable
131
of the target class and models without knowledge of the metanetwork,theeditingprocedureitselfisindependentof
132
targetclass. Whileimpressive,thismethodreliesonaccess the target model’s structure. As discussed in Section 2.1
133
totargetmodelswhichserveasagroundtruthforthede- andAppendixA,metanetworkshavebeensuccessfullyap-
134
siredperformance: modelstrainedfromscratchonadataset pliedacrossdiversearchitectures,andrecentdevelopments
135
where the data to forget is excluded. Instead, we aim to suchasgraphmetanetworks(Limetal.,2024)canhandle
136
presentamethodforeditingpre-trainedmodelswhereac- variablearchitecturalinputsbydesign.
137
cesstoaexampleeditedmodelisnotavailable,thushaving
138
improvedpracticalimplications.
139 3.Preliminaries
140
2.2.ModelEditing&MachineUnlearning Let C be a set of K classes. In the case of a model edit-
141
ingtask,theclassweaimtoinfluence,thetargetclass,is
142 Aseminalworkthatintroducedtheconceptoffindingtask
denoted as c. The set of remaining classes on which we
143 vectorsintheparameterspaceofamodelandusingthose
wishtopreserveperformancearedenotedC = C\c,the
144 toeditapre-trainedmodelisIlharcoetal.(2022a). They retain or control set. Let D = {(x ,y )}N r be a dataset
145 showthatataskorconceptcanbeunlearnedbynegatingthe i i i=1
consistingofN sample-labelpairs,whereeachdatasample
146 directioninweightspaceamodelmoveswhenfinetuning x ∈RD isassociatedwithaone-hotvectorlabely ∈RK.
147 onthatspecifictask. Theyalsoshowimprovementontasks i i
D isdefinedasthesetofdatacorrespondingtotargetclass
148 andtasktransferthroughothertask-vectorarithmetic. For lab t elc,D isthattoC . Letϕ (x):RD →RK beaneu-
149 forgettingimageclassificationtasksvianegationtheyreport ralnetwor r kparameteri r zedbyθ θ ∈ RW trainedtoclassify
150 the accuracy of a pre-trained Vision Transformer to drop
samplesbymappinginputstoclassprobabilitydistributions.
151 by45.8percentagepointsonaverageontargetclassesthey RW is called the weight space. We denote a model zoo
152 wishtoforget,withlittlelossonthecontroltask.Theyapply Φ = {ϕ (x)}M as a collection of M models sharing
153 twobaselines;finetuningviagradientascent,i.e. training θj j=1
similarmodelarchitecturebutwithvaryingparametersθ .
154 inthewrongdirection,andapplyingarandomvectorona j
155 per-layerbasisofthesamemagnitudeasthetask-vector.
156 4.Method
Post-hocmodelediting—andspecificallyunlearning—isin-
157
creasinglyimportantduetogrowingpre-trainingcostsand Inthissectionweintroducethemodel-editingalgorithm,the
158
regulatoryemphasisondataprivacyandownership. Most metanetworks,andthemodelzoousedinourexperiments.
159
unlearningmethodsassumeaccesstooriginaltrainingdata, ThecoreofourmethodisshowninAlgorithm1. Several
160
oratminimumtheforgetset(Nguyenetal.,2025). Chun- implementationsthroughasetoflossfunctionsandstopping
161
dawatetal.(2023)arguethattheseassumptionsareoften criteriaarediscussed.
162
163 1seesection8forfurtherdiscussion
164
3

UtilizingWeightSpaceLearningforData-FreeModelEditing
| 165 |     |     |     |     |     | the performance | on D is | not considered | a realistic | use |
| --- | --- | --- | --- | --- | --- | --------------- | ------- | -------------- | ----------- | --- |
r
| 166 |     |     |     |     |     | case,somerelyflippingthesignontheBoostLossseems |     |                          |     |     |
| --- | --- | --- | --- | --- | --- | ----------------------------------------------- | --- | ------------------------ | --- | --- |
| 167 |     |     |     |     |     | nonsensicalforthisexperiment.                   |     | Rather,inourlossfunction |     |     |
| 168 |     |     |     |     |     | labeledSimpleLossweonlyfocusontargetclassc:     |     |                          |     |     |
169
170
|     |          |                                             |     |     |     |     | f =ψ | (θ). |     | (2) |
| --- | -------- | ------------------------------------------- | --- | --- | --- | --- | ---- | ---- | --- | --- |
| 171 |          |                                             |     |     |     |     |      | c    |     |     |
|     | Figure2. | ThearchitectureofthemodelsintheSmallCNNZoo. |     |     |     |     |      |      |     |     |
172
Itconsistsofthreeconvolutionlayerswitha3x3kerneland16 4.3.Stoppingcriterion
173
outputchannels,followedbyaglobalaveragepoolingflattening
174
operationandafullyconnectedlayerto10outputlogits.Theinput Wecannotexpectψtobefaithfulontheentireweightspace.
| 175 | imageshaveonechannel. |     | Intotalthisresultsin4970trainable |     |     |     |     |     |     |     |
| --- | --------------------- | --- | --------------------------------- | --- | --- | --- | --- | --- | --- | --- |
Preliminaryexperimentshaveshownthatafteranumberof
176 parameters
|     |     |     |     |     |     | iterationsψcanleaveits‘trustregion’: |     |     | itcanbecomeworse |     |
| --- | --- | --- | --- | --- | --- | ------------------------------------ | --- | --- | ---------------- | --- |
177
atpredictingtheclassrecallofϕaccurately.Thuswecannot
178 4.1.AMetanetworkforData-freeEditing
relyonconvergenceofthelossasastoppingcriterionfor
179
We train a Multi-Layer Perceptron (MLP) metanetwork Algorithm1. Sincetheproposedinterventionisdata-free
180 RW RK, thereisnodirectevaluationmethodfordeterminingifand
|     | ψ(θ;ω) | : → | parameterized | by ω, | to perform |     |     |     |     |     |
| --- | ------ | --- | ------------- | ----- | ---------- | --- | --- | --- | --- | --- |
181
multivariateregressionontheper-classrecallofthemodels when the intervention is sufficiently successful. We thus
182
inΦ. Todemonstratetheusefulnessoftherepresentationψ proposeψtoserveasaproxyfordeterminingperformance
183
haslearnedofthetaskperformanceofamodelzooΦ,weat- oftheeditedmodel,asastoppingcriterionfordetermining
184
tempttoeditheld-outmodelsfromΦbasedoninformation whentohaltAlgorithm1.
185
guidedbyψ.Notabene,weperformgradientdescentonthe
186
metanetwork’sinput,ratherthanitsownparameters,i.e. the AbsoluteAccuracyPrediction Whenthepredictedac-
187
|     | weightsθoftheoriginalmodelϕ. |     |     | Thetrainedmetanetwork |     |     |     |     |     |     |
| --- | ---------------------------- | --- | --- | --------------------- | --- | --- | --- | --- | --- | --- |
curacyonthetargetclassdropsbeloworrisesaboveaset
188
ψservesasadifferentiableproxyfortheperformanceofϕ, threshold(dependingonthetask)thealgorithmishalted.
189
|     | enablingdata-freemodelediting. |     |     | Theproposedalgorithm |     |     |     |     |     |     |
| --- | ------------------------------ | --- | --- | -------------------- | --- | --- | --- | --- | --- | --- |
190
isdescribedinAlgorithm1. RelativeAccuracyPrediction Becausetheinitialaccu-
191
| 192 |     |     |     |     |     | racydiffersbetweenspecificinstancesofϕ,wemaywant |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------------------------------ | --- | --- | --- | --- |
Algorithm1WeightSpaceModelEditing
|     |     |     |     |     |     | tolookatthepredictedrelativedifference. |     |     | Forexample, |     |
| --- | --- | --- | --- | --- | --- | --------------------------------------- | --- | --- | ----------- | --- |
193
|     | Require: |     |     |     |     | wecanhaltthealgorithmoncethemetanetworkpredictsa |     |     |     |     |
| --- | -------- | --- | --- | --- | --- | ------------------------------------------------ | --- | --- | --- | --- |
194
Lossfunctionf(ψ(θ;ω)):RW→R(differentiable),
50%accuracydropovertheoriginalaccuracy.
195
|     | inputmodelϕ(x;θ)withinitialparametersθ |     |     |     | ∈RW, |                     |     |     |     |     |
| --- | -------------------------------------- | --- | --- | --- | ---- | ------------------- | --- | --- | --- | --- |
| 196 |                                        |     |     |     | 0    |                     |     |     |     |     |
| 197 | trainedmetanetworkψ(θ;ω),              |     |     |     |      | 5.Experimentalsetup |     |     |     |     |
|     | stepsizeη                              | >0, |     |     |      |                     |     |     |     |     |
198
binarystoppingcriterions(·), Toshowcasemodelediting,weperformtwospecificexperi-
199
200 toleranceε>0 mentsonpreviouslyunseenCNNs;Unlearning:decreasing
|     | 1: θ | ←θ  |     |     |     | performanceofapre-trainedϕonD |     |     | ,anduplearning: | in- |
| --- | ---- | --- | --- | --- | --- | ----------------------------- | --- | --- | --------------- | --- |
| 201 |      | 0   |     |     |     |                               |     |     | t               |     |
202 2: whilenots(θ,ε)do creasing the performance of a pre-trained ϕ on D t , both
3: g←∇ f(ψ(θ;ω)) ▷Computegradientw.r.t. input whileaimingtopreserveperformanceontheremainingdata
| 203 |             | θ       |     |               |     |     |     |     |     |     |
| --- | ----------- | ------- | --- | ------------- | --- | --- | --- | --- | --- | --- |
|     | 4:          | θ ←θ−ηg |     | ▷Gradientstep |     | D   |     |     |     |     |
| 204 |             |         |     |               |     | r   |     |     |     |     |
| 205 | 5: endwhile |         |     |               |     |     |     |     |     |     |
5.1.SmallCNNZoo
206
207
|     | 4.2.Lossfunction |     |     |     |     | ForourexperimentsweusetheSmallCNNZooasΦ(Un- |     |     |     |     |
| --- | ---------------- | --- | --- | --- | --- | ------------------------------------------- | --- | --- | --- | --- |
208
|     |                             |     |     |                    |     | terthineretal.,2021).                           | Thisdatasetcontainstrainingcheck- |     |     |     |
| --- | --------------------------- | --- | --- | ------------------ | --- | ----------------------------------------------- | --------------------------------- | --- | --- | --- |
| 209 | Forunlearning,wewanttomoveϕ |     |     | inthedirectionofan |     |                                                 |                                   |     |     |     |
|     |                             |     |     | θ                  |     | pointsfor120,000smallunderparameterizedCNNimage |                                   |     |     |     |
210
expecteddecreaseofperformanceonD t byψ(θ), andto classifiers, each trained on one of four datasets: MNIST
| 211 | maintain | performance | on D . | The loss function | used for |           |                           |     |           |       |
| --- | -------- | ----------- | ------ | ----------------- | -------- | --------- | ------------------------- | --- | --------- | ----- |
|     |          |             | r      |                   |          | (Lecun et | al., 1998), Fashion-MNIST |     | (F-MNIST) | (Xiao |
212 thistaskislabeledBoostLoss:
etal.,2017),CIFAR-10(Krizhevskyetal.,2009),orSVHN
| 213 |     |     |     |          |     | (Netzeretal.,2011). | Trainingwasdonewithawidesweep |     |     |     |
| --- | --- | --- | --- | -------- | --- | ------------------- | ----------------------------- | --- | --- | --- |
| 214 |     |     |     | (cid:88) |     |                     |                               |     |     |     |
f =ψ (θ)−β ψ (θ). (1) ofhyperparameterstoachieveazooofmodelswithawide
| 215 |     |     | c   | i   |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
rangeofvaryingperformance,resultingin4CNNcollec-
i∈Cr
216
tionseachcontainingcheckpointsofapproximately30,000
217
Fortheperformanceimprovementexperiment,wewantto CNNsover86epochs. Figure2showsavisualizationof
218 increase the performance on D , but aiming to decrease thearchitectureoftheCNNs. WepartitionΦintodisjoint
t
219
4

UtilizingWeightSpaceLearningforData-FreeModelEditing
220 setsΦ ,Φ ,Φ containing15,000,7,500,and7,500 perlayer. Theseareconcatenatedandaddedtoθ yielding
|     |     | train val | test |     |     |     |     |     |     |     |     |     | 0   |     |
| --- | --- | --------- | ---- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
221 modelsrespectively. aweight-vectorthatcanbeusedinthesamewayasθ.
222
|     | We extend | the | Small CNN | Zoo | by reconstructing |     | and re- |     |     |     |     |     |     |     |
| --- | --------- | --- | --------- | --- | ----------------- | --- | ------- | --- | --- | --- | --- | --- | --- | --- |
223
evaluatingthenetworksforclassrecall,orper-classaccu- Gradientascent AsinGolatkaretal.(2020);Tarunetal.
224
(2023);Ilharcoetal.(2022a)wecreateabaselinebyfine-
|     | racy. | Theper-classaccuracyforall120,000modelsisre- |     |     |     |     |     |     |     |     |     |     |     |     |
| --- | ----- | -------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
225
portedforeachepochstageinthecanonicalsplit(early,mid- tuningexclusivelyonthetargettask,butnegatingtheloss
226
dle,final)andmadeavailableat[redactedforanonymity]. function. Effectivelyunlearningviagradientascent.
227
228 Fortheunlearningexperimentwefilteroutextremelypoor
|     |     |     |     |     |     |     |     | Retain-setfinetuning |     |     | Forunlearning,astandardapprox- |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------- | --- | --- | ------------------------------ | --- | --- | --- |
229 performingorincompletemodelsfromtheevaluationset.
|     |     |     |     |     |     |     |     | imate unlearning |     | baseline | is finetuning |     | on the retain | set |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | --- | -------- | ------------- | --- | ------------- | --- |
230 Allmodelsthathave0%accuracyinanyclassorlessthan
D (alldataexcepttheforgetclass)forafewepochs(Go-
| 231 | 10%accuracyinthetargetclassareexcludedfromtheeval- |     |     |     |     |     |     | r                                    |     |     |     |     |                 |     |
| --- | -------------------------------------------------- | --- | --- | --- | --- | --- | --- | ------------------------------------ | --- | --- | --- | --- | --------------- | --- |
| 232 |                                                    |     |     |     |     |     |     | latkaretal.,2020;Fosteretal.,2024b). |     |     |     |     | FollowingFoster |     |
uationset.
|     |     |     |     |     |     |     |     | etal.(2024b), |     | weuse5epochswiththeoriginaltraining |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ----------------------------------- | --- | --- | --- | --- |
233
hyperparameters.
234
5.2.Metanetworkdetails
235
|     | ψtakestheflattenedCNNweightvectorθasinput. |     |     |     |     |     | Thenet- | 5.5.Experiments |     |     |     |     |     |     |
| --- | ------------------------------------------ | --- | --- | --- | --- | --- | ------- | --------------- | --- | --- | --- | --- | --- | --- |
236
237 workconsistsof5hiddenlayerswith256ReLU-activated
|     |                                 |     |     |     |     |                  |     | We trained                         | 10  | separate | metanetworks | per | to predict | per- |
| --- | ------------------------------- | --- | --- | --- | --- | ---------------- | --- | ---------------------------------- | --- | -------- | ------------ | --- | ---------- | ---- |
|     | unitseachandadropoutrateof0.03. |     |     |     |     | Thefinallayerisa |     |                                    |     |          |              |     |            |      |
| 238 |                                 |     |     |     |     |                  |     | classrecallperallfourΦcollections. |     |          |              |     |            |      |
lineartransformationfollowedbyasigmoidactivationto Wereporttheaverage
239
|     |                                      |     |     |     |                     |                  |     | coefficient                                          | of determination |     | R2 in | addition | to MSE | and |
| --- | ------------------------------------ | --- | --- | --- | ------------------- | ---------------- | --- | ---------------------------------------------------- | ---------------- | --- | ----- | -------- | ------ | --- |
|     | producemulti-outputregressionvalues. |     |     |     |                     | Thesehyperparam- |     |                                                      |                  |     |       |          |        |     |
| 240 |                                      |     |     |     |                     |                  |     | MAEforthismultivariateregressiontask,asinUnterthiner |                  |     |       |          |        |     |
|     | eterswerechosentomaximizeR2          |     |     |     | onavalidationsetand |                  |     |                                                      |                  |     |       |          |        |     |
241
|     |         |              |       |                             |         |           |            | et al. (2021). | One      | of                                 | the 10 metanetworks |      | is selected | at    |
| --- | ------- | ------------ | ----- | --------------------------- | ------- | --------- | ---------- | -------------- | -------- | ---------------------------------- | ------------------- | ---- | ----------- | ----- |
| 242 | were    | not found    | to be | too sensitive               | to      | tuning.   | We include |                |          |                                    |                     |      |             |       |
|     |         |              |       |                             |         |           |            | random,        | and used | to                                 | edit models         | of Φ | , for every | class |
|     | CNN     | checkpoints  | from  | early,                      | middle, | and final | training   |                |          |                                    |                     | test |             |       |
| 243 |         |              |       |                             |         |           |            | between0-9asc. |          | Forallexperimentsastepsizeηof0.1is |                     |      |             |       |
|     | stages. | Thisexposesψ |       | toabroaderrangeofweightcon- |         |           |            |                |          |                                    |                     |      |             |       |
244
chosen.
figurations,whichwefindmarginallyimprovesprediction
245
faithfulnesswhenweightsaremodifiedduringediting. Runsthatdonotterminateafter2000steps,aswellasruns
246
| 247 |     |     |     |     |     |     |     | thatdonotmovethemodelmorethan1arb. |     |     |     |     | unitEuclidean |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------------------------- | --- | --- | --- | --- | ------------- | --- |
5.3.Scorefunctions distance in weight space are excluded from the reported
248
249 results. These post-hoc filters sort out models where the
LikeIlharcoetal.(2022a)wereportaccuracyonatarget
methodwasclearlyineffective,notinapositiveornegative
| 250 | (D )andtheaverageaccuracyoncontroldataset(D |     |     |     |     |     | ). In |        |     |     |     |     |     |     |
| --- | ------------------------------------------- | --- | --- | --- | --- | --- | ----- | ------ | --- | --- | --- | --- | --- | --- |
|     | t                                           |     |     |     |     |     | r     | sense. |     |     |     |     |     |     |
251
addition,wemeasurethesuccessoftheinterventionsby2
252
|     | differentscores: |     | Targetdifference-Directchangeinclass |     |     |     |     |     |     |     |     |     |     |     |
| --- | ---------------- | --- | ------------------------------------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
253 accuracyofclassc. Maxdifference-Changeoftheclassc Unlearning For unlearning we aim to reduce accuracy
254 on a target class whilst retaining accuracy on the control
|     | relativetothemaximumchangeoftheclassesinC |     |     |     |     |     | . This |     |     |     |     |     |     |     |
| --- | ----------------------------------------- | --- | --- | --- | --- | --- | ------ | --- | --- | --- | --- | --- | --- | --- |
| 255 |                                           |     |     |     |     |     | r      |     |     |     |     |     |     |     |
isastrictmetric: ifatleastoneoftheuntargetedclasseshas classes. AsalossfunctionweusetheBoostLossfunction
| 256 |       |        |             |         |     |               |         | (Equation1)withβ                                   |     | =0.1andtherelativeaccuracypredic- |     |     |     |     |
| --- | ----- | ------ | ----------- | ------- | --- | ------------- | ------- | -------------------------------------------------- | --- | --------------------------------- | --- | --- | --- | --- |
|     | had a | change | larger than | that of | the | target class, | the max |                                                    |     |                                   |     |     |     |     |
| 257 |       |        |             |         |     |               |         | tionstoppingcriterionatwithathresholdat60%accuracy |     |                                   |     |     |     |     |
differencescorewillbenegative.
| 258 |     |     |     |     |     |     |     | reduction. | MotivationforthisisexplainedintheSection6. |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ------------------------------------------ | --- | --- | --- | --- | --- |
259
5.4.Baselines
260
|     |            |         |       |           |     |           |             | Uplearning | Foruplearningweaimtoimproveaccuracy |     |     |     |     |     |
| --- | ---------- | ------- | ----- | --------- | --- | --------- | ----------- | ---------- | ----------------------------------- | --- | --- | --- | --- | --- |
| 261 | We compare | against | three | baselines |     | that test | whether the |            |                                     |     |     |     |     |     |
onthetargetclasswhilstretainingaccuracyonthecontrol
| 262 | metanetworkprovidesameaningfuleditingsignal: |     |     |     |     |     | aran- |     |     |     |     |     |     |     |
| --- | -------------------------------------------- | --- | --- | --- | --- | --- | ----- | --- | --- | --- | --- | --- | --- | --- |
classes. WeusetheSimpleLossfunction(Equation2)and
263 dom edit of the same magnitude as our edits, and for the theabsoluteaccuracypredictionstoppingcriterionat90%.
264 unlearningexperimentanaivefinetuningsolutionandan
Thisthresholdissethightomaximizeuplearningeffortsfor
| 265 | idealbaselineusingfinetuningontheretainsetD |     |     |     |     |     | .   |                |     |     |     |     |     |     |
| --- | ------------------------------------------- | --- | --- | --- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- |
|     |                                             |     |     |     |     |     | r   | demonstration. |     |     |     |     |     |     |
266
267
6.Results
| 268 | Random      | vector | Like     | Ilharco      | et al. | (2022a) | we evalu-     |     |     |     |     |     |     |     |
| --- | ----------- | ------ | -------- | ------------ | ------ | ------- | ------------- | --- | --- | --- | --- | --- | --- | --- |
| 269 | ate against | a      | baseline | of per-layer |        | random  | vector edits. |     |     |     |     |     |     |     |
Wefindconsistenthighperformanceforpredictingperclass
| 270 | We compute |     | the per | layer magnitude |     | of our | proposed |           |     |                                  |     |     |     |     |
| --- | ---------- | --- | ------- | --------------- | --- | ------ | -------- | --------- | --- | -------------------------------- | --- | --- | --- | --- |
|     |            |     |         |                 |     |        |          | recallofΦ | .   | TheseresultsarereportedinTable1. |     |     |     | The |
t es t
| 271 | edit τ(L)                                        | = ||θ(L) | −   | θ (L)||, then | draw | a random | vector |            |                                               |     |     |     |     |     |
| --- | ------------------------------------------------ | -------- | --- | ------------- | ---- | -------- | ------ | ---------- | --------------------------------------------- | --- | --- | --- | --- | --- |
|     |                                                  |          |     | 0             |      |          |        | resultingm | e t anetworksfunctionhighlyconsistent,anerror |     |     |     |     |     |
| 272 | υ(L) ∼N(0,I)fromastandardGaussiandistributionand |          |     |               |      |          |        |            |                                               |     |     |     |     |     |
marginof±1standarddeviationoverthe10trialsforthe
273 scaleittoτ(L),givingτ(L) = υ(L) τ(L),arandomedit R2isreported.
|     |     |     |     | random | ||υ(L)|| |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ------ | -------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
274
5

UtilizingWeightSpaceLearningforData-FreeModelEditing
| 275 |     |     |       |     |            |     |     |      | MNIST |     |     |      | Fashion-MNIST |     |
| --- | --- | --- | ----- | --- | ---------- | --- | --- | ---- | ----- | --- | --- | ---- | ------------- | --- |
|     |     |     | train |     | validation |     |     | 1.00 |       |     |     | 1.00 |               |     |
276
|     |     |     |     |     |     |     |     | 0.75 |     |     |     | 0.75 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | ---- | --- | --- |
R2
277 dataset mse mae mse mae ecnereffiD tegraT 0.50 ecnereffiD tegraT 0.50
| 278 |       |     |       |             |       |             |     | 0.25 |     |     |     | 0.25 |     |     |
| --- | ----- | --- | ----- | ----------- | ----- | ----------- | --- | ---- | --- | --- | --- | ---- | --- | --- |
|     | MNIST |     | 0.012 | 0.050 0.014 | 0.056 | 0.922±0.001 |     |      |     |     |     |      |     |     |
| 279 |       |     |       |             |       |             |     | 0.00 |     |     |     | 0.00 |     |     |
0.893±0.003
|     | F-MNIST |     | 0.013 | 0.060 0.016 | 0.069 |     |     | 0.25 |     |     |     | 0.25 |     |     |
| --- | ------- | --- | ----- | ----------- | ----- | --- | --- | ---- | --- | --- | --- | ---- | --- | --- |
280
|     | CIFAR-10 |     | 0.012 | 0.062 0.015 | 0.076 | 0.696±0.012 |     | 0.50 |     |     |     | 0.50 |     |     |
| --- | -------- | --- | ----- | ----------- | ----- | ----------- | --- | ---- | --- | --- | --- | ---- | --- | --- |
281
|     | SVHN |     | 0.009 | 0.047 0.012 | 0.055 | 0.855±0.025 |     |      |     |     |     |      |     |     |
| --- | ---- | --- | ----- | ----------- | ----- | ----------- | --- | ---- | --- | --- | --- | ---- | --- | --- |
|     |      |     |       |             |       |             |     | 0.75 |     |     |     | 0.75 |     |     |
282
Table1.Averageof10metanetworksperformanceforpredicting 1.00 1.00
| 283 |     |     |     |     |     |     |     | 9   | 1 2 4 5 | 7 0 8 3 6 |     | 5 4 | 7 9 2 | 0 3 1 6 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------- | --------- | --- | --- | ----- | --------- |
classrecallonvalidationsetsperimage-dataset. Uncertaintyre- Target Class Target Class
284
|     | portedforR2. |     |     |     |     |     |     | 1.00 | CIFAR-10 |     |     | 1.00 | SVHN |     |
| --- | ------------ | --- | --- | --- | --- | --- | --- | ---- | -------- | --- | --- | ---- | ---- | --- |
285
|     |     |     |     |     |     |     |     | 0.75 |     |     |     | 0.75 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | ---- | --- | --- |
286
|     |     |     |     |     |     |     |     | ecnereffiD tegraT 0.50 |     |     | ecnereffiD tegraT | 0.50 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------------- | --- | --- | ----------------- | ---- | --- | --- |
287
|     | Anunbiasedsampleofresultingpredictionsonindividual |     |            |              |     |                   |     | 0.25 |     |     |     | 0.25 |     |     |
| --- | -------------------------------------------------- | --- | ---------- | ------------ | --- | ----------------- | --- | ---- | --- | --- | --- | ---- | --- | --- |
| 288 |                                                    |     |            |              |     |                   |     | 0.00 |     |     |     | 0.00 |     |     |
|     | models,                                            | and | the effect | of Algorithm |     | 1 for unlearning, | is  |      |     |     |     |      |     |     |
| 289 |                                                    |     |            |              |     |                   |     | 0.25 |     |     |     | 0.25 |     |     |
reportedinAppendixD.
|     |     |     |     |     |     |     |     | 0.50 |     |     |     | 0.50 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | ---- | --- | --- |
290
|     |     |     |     |     |     |     |     | 0.75 |     |     |     | 0.75 |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---- | --- | --- | --- | ---- | --- | --- |
291
|     | 6.1.Unlearning |         |         |                   |     |            |     | 1.00 |              |           |     | 1.00 |              |           |
| --- | -------------- | ------- | ------- | ----------------- | --- | ---------- | --- | ---- | ------------ | --------- | --- | ---- | ------------ | --------- |
| 292 |                |         |         |                   |     |            |     | 8    | 1 0 5 9      | 4 6 7 2 3 |     | 1 2  | 4 7 3        | 9 6 8 0 5 |
|     |                |         |         |                   |     |            |     |      | Target Class |           |     |      | Target Class |           |
| 293 | Figure         | 3 shows | results | of an exploratory |     | experiment | for |      |              |           |     |      |              |           |
Figure4.Boxplotsoftargetdifferencebytargetclassafterinter-
| 294 | tuningrelativestoppingthreshold. |     |     |     | Unlearningisattempted |     |     |     |     |     |     |     |     |     |
| --- | -------------------------------- | --- | --- | --- | --------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
ventionforMNIST,Fashion-MNIST,CIFAR-10,andSVHNon
295 for 1494 MNIST validation set models on target class 4, thevalidationset.Orderedbymeantargetdifference;outliersomit-
296 varyingrelativestoppingthreshold. Thisissetoutagainsta ted.Upwardsofthereddottedlinemarkssuccessfulintervention.
297
|     | baselineofrandomvectoreditsofthesamemagnitude. |     |     |     |     |     | We  |     |     |     |     |     |     |     |
| --- | ---------------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
thetargetclassmoveswithatleastoneotherclass.
298 findthatincreasingstoppingthresholdincreasestheeffect
299 ofourinterventionontargetclass,butalsodeterioratesthe
Table2reportsmedianunlearningresultsaggregatedover
300 controlclasses.Tobalancetheseeffectswechoosearelative all10separatetargetclasses. Thegradientascentbaseline
301 stoppingthresholdof60%forfurtherexperiments. achievesthelowesttargetrecall(D )butcatastrophically
t
302
|     |                                                       |     |     |     |     |     |     | damagescontrolgroup(D                           |     |     | )performance. |     | Therandomvec- |     |
| --- | ----------------------------------------------------- | --- | --- | --- | --- | --- | --- | ----------------------------------------------- | --- | --- | ------------- | --- | ------------- | --- |
|     | Afterevaluation,runsthatdidnotterminateafter2000steps |     |     |     |     |     |     |                                                 |     | r   |               |     |               |     |
| 303 |                                                       |     |     |     |     |     |     | toreditsshowthatrandomeditsofthesamemagnitudeas |     |     |               |     |               |     |
andrunsthatdidnotmovethemodelsignificantlyinweight
304
space. i.e. lessthan1a.u. Euclideandistance,areexcluded ourproposedhaveaslightanduniformdeterioratingeffect
| 305 |     |     |     |     |     |     |     | overbothtargetandcontrolgroups. |     |     |     | Ourmethodbalances |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------- | --- | --- | --- | ----------------- | --- | --- |
formtheresultsreportedhere.
| 306 |     |     |     |     |     |     |     | targetgroupreductionwithcontrolgrouppreservation. |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
307 Boxplotsofthetargetdifferenceandmaxdifferencescores
308 perclassarereportedinFigure4andFigure5respectively. 6.2.Uplearning
309 Inalmostallcasesamedianpositiveresultoftheinterven-
| 310 |                 |     |                                   |     |     |     |     | Table3reportsmedianuplearningresultsaggregatedover |     |     |     |     |     |     |
| --- | --------------- | --- | --------------------------------- | --- | --- | --- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|     | tionisachieved. |     | Thisisevidentbyanetpositivemedian |     |     |     |     |                                                    |     |     |     |     |     |     |
311 targetdifference,greenlinesintheboxplots. Theinterven- all10separateclassesperdataset. Againwefindthatthe
312 tionisnotalwaysisolated. Figure5showsthatonaverage interventiononaverageissuccessfulforalldatasets,except
forSVHN.OntheSVHNmodelsourmethodperformssimi-
313
| 314 |     |                                          |     |               |     |     |     | lartoarandomvectoredit.                            |     |     | Therelativelypoorperformance |     |     |     |
| --- | --- | ---------------------------------------- | --- | ------------- | --- | --- | --- | -------------------------------------------------- | --- | --- | ---------------------------- | --- | --- | --- |
| 315 |     |                                          |     |               |     |     |     | mightbeexplainedbyanoverlyoptimisticstoppingcrite- |     |     |                              |     |     |     |
|     |     | 50                                       |     |               |     | 1.0 |     |                                                    |     |     |                              |     |     |     |
|     |     | )%( pord ycarucca tegrof tegrat evitaler |     | random vector |     |     |     |                                                    |     |     |                              |     |     |     |
| 316 |     |                                          |     |               |     |     |     | rion. MostCIFAR-10andSVHNmodelsintheSmallCNN       |     |     |                              |     |     |     |
our method
| 317 |     | 40  |     |     |     |     |     | Zoo dataset | have | low overall | performance, |     | which | might |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ---- | ----------- | ------------ | --- | ----- | ----- |
0.8
make90%performanceonaclassunreachable.
318
| 319 |     | 30  |     |     |     | 0.6 dlohserht pots |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
320
7.Discussion
| 321 |     | 20  |     |     |     | 0.4 |     |                                                     |             |      |              |     |                 |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------------------------------------------- | ----------- | ---- | ------------ | --- | --------------- | --- |
| 322 |     |     |     |     |     |     |     | WefindthatCNNscanbeeditedbyapplyingeditsobtained    |             |      |              |     |                 |     |
|     |     |     |     |     |     |     |     | via input                                           | attribution | of a | metanetwork. |     | Our experiments |     |
| 323 |     | 10  |     |     |     | 0.2 |     |                                                     |             |      |              |     |                 |     |
| 324 |     |     |     |     |     |     |     | highlightthepotentialofeditingmodelsbyapplyingtask- |             |      |              |     |                 |     |
325 0 vectors obtained via backpropagation of an arbitrary loss
0.0
0 10 20 30 40 50 functionwithrespecttotheinputofametanetwork.
| 326 |     |     | avg relative retain accuracy drop (%) |     |     |     |     |     |     |     |     |     |     |     |
| --- | --- | --- | ------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
327
Figure3.ForgetclassversusRetainclassesmedianrelativeaccu- Figure4and5showalargespreadintheeffectivenessof
328 racydrop(%)for1494MNISTvalidationsetmodels. ourmethodssofar. Wefindthatourmethodhasavarying
329
6

UtilizingWeightSpaceLearningforData-FreeModelEditing
Table2.Unlearningbaselinecomparisonacrossdatasets.Medianaccuracyoveralltargetclassesafterintervention.Target(D )recall
| 330 |     |     |     |     |     |     |     |     |     | t   |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
shoulddecrease;Control(D )recallshouldremainhigh.Numberofalgorithmevaluations:MNIST:27021,F-MNIST:32086,CIFAR-10:
| 331 |     |     | r   |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
26952,SVHN:13838.
332
333
|     |     |     |     |     | MNIST | F-MNIST | CIFAR-10 | SVHN |     |     |
| --- | --- | --- | --- | --- | ----- | ------- | -------- | ---- | --- | --- |
334
|     |     |     | Method | D   | ↓ D | ↑ D ↓ D | ↑ D ↓ | D ↑ D ↓ | D ↑ |     |
| --- | --- | --- | ------ | --- | --- | ------- | ----- | ------- | --- | --- |
|     |     |     |        |     | t   | r t     | r t   | r t     | r   |     |
335
|     |     |     | Before | 0.87 | 0.89 | 0.78 0.75 | 0.42 | 0.38 0.48 | 0.43 |     |
| --- | --- | --- | ------ | ---- | ---- | --------- | ---- | --------- | ---- | --- |
336
|     |     |     | Retainfinetune | 0.01 | 0.89 | 0.00 0.77 | 0.00 | 0.40 0.00 | 0.45 |     |
| --- | --- | --- | -------------- | ---- | ---- | --------- | ---- | --------- | ---- | --- |
337
|     |     |     | Gradientascent | 0.00 | 0.13 | 0.00 0.15 | 0.00 | 0.12 0.00 | 0.11 |     |
| --- | --- | --- | -------------- | ---- | ---- | --------- | ---- | --------- | ---- | --- |
338
| 339 |     |     | Randomvector | 0.82 | 0.82 | 0.74 0.72 | 0.35 | 0.33 0.36 | 0.35 |     |
| --- | --- | --- | ------------ | ---- | ---- | --------- | ---- | --------- | ---- | --- |
|     |     |     | Ourmethod    | 0.61 | 0.73 | 0.36 0.65 | 0.13 | 0.34 0.29 | 0.35 |     |
340
341
342 Table3.Resultsforuplearning.Medianaccuracyoveralltargetclassesafterintervention.Target(D )recallshouldincrease;Control
t
(D )recallshouldremain.Numberofalgorithmevaluations:MNIST:2632,F-MNIST:1239,CIFAR-10:8841,SVHN:2214.
343 r
344
|     |     |     |     |     | MNIST | F-MNIST | CIFAR-10 | SVHN |     |     |
| --- | --- | --- | --- | --- | ----- | ------- | -------- | ---- | --- | --- |
345
|     |     |     | Method       | D    | ↑ D  | ↓ D ↑ D   | ↓ D ↑ | D ↓ D ↑   | D ↓  |     |
| --- | --- | --- | ------------ | ---- | ---- | --------- | ----- | --------- | ---- | --- |
| 346 |     |     |              |      | t r  | t r       | t     | r t       | r    |     |
| 347 |     |     | Before       | 0.69 | 0.72 | 0.70 0.74 | 0.36  | 0.34 0.39 | 40   |     |
| 348 |     |     | Randomvector | 0.64 | 0.65 | 0.66 0.67 | 0.18  | 0.24 0.17 | 0.24 |     |
| 349 |     |     | Ourmethod    | 0.72 | 0.62 | 0.81 0.57 | 0.61  | 0.19 0.20 | 0.23 |     |
350
351
352 effectpermodel,andthereisabiasbetweenspecificclasses: unlearningalgorithmissolelybasedonalossfunctionon
353 someclassesareeffectedbetterthanothers. Inpracticethis themetanetworkpredictions. Thismeansthatmetanetwork-
354 meansthattherearemanyexamplesforwhichmetanetwork- basedmodeleditingcanbeatrulydata-freemethodology.
based model editing works very well (Figure 1 (bottom) This makes the approach applicable in realistic settings
355
356 showsexamplewheretheunlearningeditwasexceptionally whereoriginaltrainingdataareunavailable,restricted,or
357 effective), buttherearealsoagoodnumberofmodelsin legallyprotected.
| 358 | Ψ test onwhichthemethoddoesnotworkorevenshowsa |                                        |     |     |     |     |     |     |     |     |
| --- | ---------------------------------------------- | -------------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- |
| 359 | negativeeffect.                                | Furthermore,earlyexperimentationshowed |     |     |     |     |     |     |     |     |
8.Outlook
thatwhilenothavingasignificanteffectonthepopulation-
360
361 levelresults,theresultsofmetanetwork-basedmodelediting Thisworkismainlypresentedasaproofofconcept,anon-
362 coulddifferlargelybasedonspecificϕ-ψcombinations. As exhaustive list of potential promising avenues for future
363 ofyetitisunclearwhatcausestheinter-modelvariance,and workarediscussedhere.
| 364 | theinter-classbias.                 |     | Furtherresearchisneededtointerpret |                  |     |                     |     |     |                         |     |
| --- | ----------------------------------- | --- | ---------------------------------- | ---------------- | --- | ------------------- | --- | --- | ----------------------- | --- |
| 365 | andexplainthesourcesofthisvariance. |     |                                    | Itmightpartially |     |                     |     |     |                         |     |
|     |                                     |     |                                    |                  |     | Singlemodelapproach |     |     | Thesamemethodologymaybe |     |
366 beexplainedbythenatureoftheSmallCNNModelZoo. used to edit single models locally. Metanetwork training
367 Themodelsinthismodelzooweretrainedwithawidearray
couldbedoneonadatasetbasedonnoiseperturbationof
368 ofvaryinghyperparameters,somewelloutsidetherangeof amodelparameters. Ametanetworkcouldthenbeusedto
369 commonvaluesforeffectivetraining. Thiswasdonewith editthatspecificmodelusingmethodologydescribedhere.
370 theexplicitpurposeofcreatingaverydiversepopulationof
371 CNNmodels. Althoughthisisfavorablefortrainingrobust Morefine-grainedconcepts Inthiswork, weusedper
| 372 | metanetworks | which | have seen a wide | range | of different |     |     |     |     |     |
| --- | ------------ | ----- | ---------------- | ----- | ------------ | --- | --- | --- | --- | --- |
classaccuracyasthepredictorvariableofthemetanetwork.
models(asdiscussedinSection5.2),futureworkcouldlook
| 373 |               |      |                        |     |                 | Thisshowsthatmetanetworksmaybeusedtopredictmodel |     |     |     |     |
| --- | ------------- | ---- | ---------------------- | --- | --------------- | ------------------------------------------------ | --- | --- | --- | --- |
| 374 | into creating | more | controlled populations |     | of test models, |                                                  |     |     |     |     |
propertiesorfunctionalitythatarenotdirectlyevidentfrom
| 375 | for better | interpretable | experimentation. | The | CNNs | are                           |     |     |                      |     |
| --- | ---------- | ------------- | ---------------- | --- | ---- | ----------------------------- | --- | --- | -------------------- | --- |
|     |            |               |                  |     |      | simpleinspectionoftheweights. |     |     | Thismethodologycould |     |
376 alsowellintheunderparameterizedregimewithonly4970 beappliedinanevenmorefine-grainedmatteraslongas
| 377 | trainableparameters. |     | Thiscouldbeacauseforthedifficulty |     |     |      |           |           |            |                     |
| --- | -------------------- | --- | --------------------------------- | --- | --- | ---- | --------- | --------- | ---------- | ------------------- |
|     |                      |     |                                   |     |     | some | predictor | value may | be defined | and evaluated, e.g. |
ofisolatingeditstoaspecificclassasshowninFigure5.
| 378 |     |     |     |     |     | predictingconfusionbetweenclassesorperformanceona |     |     |     |     |
| --- | --- | --- | --- | --- | --- | ------------------------------------------------- | --- | --- | --- | --- |
379
Crucially,boththeunlearningprocedureandthemetanet- specificsubsetofthedata. Thiscouldpotentiallyallowfor
380
worktrainingcanbeachievedwithoutaccesstoCNNinput metanetworkstobeusedforde-biasingmodels.
381
|     | data. | For the metanetwork | training | a dataset | of trained |     |     |     |     |     |
| --- | ----- | ------------------- | -------- | --------- | ---------- | --- | --- | --- | --- | --- |
382
modelsandtheirevaluationstatisticsissufficient,andthe Othermodelzoos Novelinsightsanddedicatedefforts
| 383 |     |     |     |     |     | have | driven | the development | of larger | model zoos cover- |
| --- | --- | --- | --- | --- | --- | ---- | ------ | --------------- | --------- | ----------------- |
384
7

UtilizingWeightSpaceLearningforData-FreeModelEditing
385 MNIST Fashion-MNIST Privacy-preservingauditing. WeightSpacemodeledit-
|     |     | 1.00 |     |     |     | 1.00 |     |                                                  |     |     |     |     |     |
| --- | --- | ---- | --- | --- | --- | ---- | --- | ------------------------------------------------ | --- | --- | --- | --- | --- |
| 386 |     |      |     |     |     |      |     | inghasthepotentialtoopenanewoperationalparadigm: |     |     |     |     |     |
|     |     | 0.75 |     |     |     | 0.75 |     |                                                  |     |     |     |     |     |
387 0.50 0.50 Third-partymodelediting. Aregulatororauditorcannow
|     | ecnereffiD xaM |     |     |     | ecnereffiD xaM |     |     |     |     |     |     |     |     |
| --- | -------------- | --- | --- | --- | -------------- | --- | --- | --- | --- | --- | --- | --- | --- |
388 0.25 0.25 receiveaproprietarymodel, unlearnadiscoveredbiasor
389 0.00 0.00 vulnerability using a pretrained metanetwork, and return
|     |     | 0.25 |              |           |     | 0.25 |                 |                                                       |     |     |     |                      |     |
| --- | --- | ---- | ------------ | --------- | --- | ---- | --------------- | ----------------------------------------------------- | --- | --- | --- | -------------------- | --- |
| 390 |     |      |              |           |     |      |                 | thesanitizedmodelwithoutthemodelownereverdisclos-     |     |     |     |                      |     |
|     |     | 0.50 |              |           |     | 0.50 |                 |                                                       |     |     |     |                      |     |
| 391 |     |      |              |           |     |      |                 | ingtheirproprietarytrainingdata.                      |     |     |     | Thisisimpossiblewith |     |
|     |     | 0.75 |              |           |     | 0.75 |                 |                                                       |     |     |     |                      |     |
| 392 |     |      |              |           |     |      |                 | gradient-basedunlearningmethodsthatrequiredataaccess. |     |     |     |                      |     |
|     |     | 1.00 |              |           |     | 1.00 |                 |                                                       |     |     |     |                      |     |
| 393 |     | 1    | 9 4 5 8      | 2 7 6 0 3 |     | 7 4  | 5 2 6 9 0 3 1 8 |                                                       |     |     |     |                      |     |
|     |     |      | Target Class |           |     |      | Target Class    |                                                       |     |     |     |                      |     |
394
|     |     | 1.00 | CIFAR-10 |     |     | 1.00 | SVHN |     |     |     |     |     |     |
| --- | --- | ---- | -------- | --- | --- | ---- | ---- | --- | --- | --- | --- | --- | --- |
395
|     |     | 0.75 |     |     |     | 0.75 |     |     |     |     |     |     |     |
| --- | --- | ---- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
396
|     | ecnereffiD xaM | 0.50 |     |     | ecnereffiD xaM | 0.50 |     |     |     |     |     |     |     |
| --- | -------------- | ---- | --- | --- | -------------- | ---- | --- | --- | --- | --- | --- | --- | --- |
397
|     |     | 0.25 |     |     |     | 0.25 |     |                         |     |     |                              |     |     |
| --- | --- | ---- | --- | --- | --- | ---- | --- | ----------------------- | --- | --- | ---------------------------- | --- | --- |
| 398 |     | 0.00 |     |     |     | 0.00 |     |                         |     |     |                              |     |     |
|     |     |      |     |     |     |      |     | Legacymodelretrofitting |     |     | Inpracticaldeeplearninglife- |     |     |
| 399 |     | 0.25 |     |     |     | 0.25 |     |                         |     |     |                              |     |     |
cycles,trainedmodelsfrequentlyoutlivetheirunderlying
|     |     | 0.50 |     |     |     | 0.50 |     |     |     |     |     |     |     |
| --- | --- | ---- | --- | --- | --- | ---- | --- | --- | --- | --- | --- | --- | --- |
400
0.75 0.75 datasets. As Chundawat et al. (2023) observe, the avail-
401
1.00 1.00 abilityoftrainingdataisoftentransienti.e. constrainedby
| 402 |     | 8   | 1 0 5 9 | 6 4 7 3 2 |     | 4 1 | 3 7 9 2 6 5 8 0 |     |     |     |     |     |     |
| --- | --- | --- | ------- | --------- | --- | --- | --------------- | --- | --- | --- | --- | --- | --- |
Target Class Target Class limited-duration cloud licenses, evolving access rights in
403
Figure5. Boxplotsofmaxdifferencebetweentargetclassand datamarketplaces,orstrictprivacymandatesliketheGDPR.
404
allotherclassesafterinterventionforMNIST,Fashion-MNIST, Thusorganizationscanrelyon’orphaned’models: valuable
| 405 | CIFAR-10,andSVHNonthevalidationset. |     |     |     |     |     | Orderedbymean |     |     |     |     |     |     |
| --- | ----------------------------------- | --- | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- |
predictiveassetswhosetrainingdataisnotavailableany-
406 targetdifference;outliersomitted.Upwardsofthereddottedline
more. Whenregulationsorusersmandateconceptremoval,
407 markssuccessfulintervention.
thesemodelsfaceobsolescence,asretrainingisimpossible.
408
|     |     |     |     |     |     |     |     | Our framework |     | can resolve | this | by enabling | compliance |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | ----------- | ---- | ----------- | ---------- |
409
|     |     |     |     |     |     |     |     | retrofitting | directly | in  | weight | space. This | allows for the |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | -------- | --- | ------ | ----------- | -------------- |
410 ingmultiplearchitectures,whichmaybemoresuitablefor sanitizationoflegacyassetswithouttheneedforlostdata,
411 robustweightspacelearning(Schu¨rholtetal.,2022b)and
oranattempttoreconstructit.
| 412 | therefore |        | can aid | in effective | model       | editing | by metanet-     |     |     |     |     |     |     |
| --- | --------- | ------ | ------- | ------------ | ----------- | ------- | --------------- | --- | --- | --- | --- | --- | --- |
| 413 | works.    | Future | work    | should       | investigate |         | how larger zoos |     |     |     |     |     |     |
9.Conclusion
414 anddifferentinputmodelarchitecturescanimproveperfor-
415 mance,generalization,andstabilityinweight-spacemodel Weintroduceddata-freemodeleditingthroughthelensof
416 editing.
WSL.Bytrainingametanetworktoaccuratelypredictthe
417
per-classrecallofamodeldirectlyfromitsparameters,we
418
Othermetanetworkarchitectures Inthisworkweuseda establishedadifferentiableproxyformodelperformance
419
|     |                                        |     |     |     |     |     |            | that requires | no  | access | to the | original training | or forget |
| --- | -------------------------------------- | --- | --- | --- | --- | --- | ---------- | ------------- | --- | ------ | ------ | ----------------- | --------- |
| 420 | simpleMLPmetanetworkasaproofofconcept. |     |     |     |     |     | Forediting |               |     |        |        |                   |           |
data. OurempiricalresultsontheSmallCNNZoodemon-
|     | larger | model | architectures |     | a more | sophisticated | metanet- |     |     |     |     |     |     |
| --- | ------ | ----- | ------------- | --- | ------ | ------------- | -------- | --- | --- | --- | --- | --- | --- |
421
stratethatbackpropagatingatask-specificlossthroughthis
| 422 | work | architecture |     | may be | required. | E.g. | CNNs, Neural |     |     |     |     |     |     |
| --- | ---- | ------------ | --- | ------ | --------- | ---- | ------------ | --- | --- | --- | --- | --- | --- |
metanetworkallowsfortargetedinterventions,effectively
|     | Functional |     | Networks | (NFNs) | (Zhou | et al., | 2023), Trans- |     |     |     |     |     |     |
| --- | ---------- | --- | -------- | ------ | ----- | ------- | ------------- | --- | --- | --- | --- | --- | --- |
423
formers(Knyazevetal.,2023),orgraphmetanetworks(Lim unlearningspecificclassesorimprovingperformancewhile
424
largelypreservingfunctionalityoncontroltasks.
425 etal.,2024)
| 426 |     |     |     |     |     |     |     | Thismethodologyoffersapotentialsolutionformanaging |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------------------------------------- | --- | --- | --- | --- | --- |
427 models where training data is restricted, lost, or legally
|     | Dealingwithsymmetries |     |     |     | Othermetanetworkarchitec- |     |     |            |     |          |             |         |              |
| --- | --------------------- | --- | --- | --- | ------------------------- | --- | --- | ---------- | --- | -------- | ----------- | ------- | ------------ |
|     |                       |     |     |     |                           |     |     | protected. | By  | enabling | third-party | editing | and privacy- |
428
turesmightalsohelpindealingwiththemanysymmetries
| 429 |      |        |              |     |            |       |             | preservingauditing,ourapproachcircumventsthelimita-     |     |     |     |     |     |
| --- | ---- | ------ | ------------ | --- | ---------- | ----- | ----------- | ------------------------------------------------------- | --- | --- | --- | --- | --- |
|     | (CNN | filter | permutation, |     | MLP hidden | layer | permutation |                                                         |     |     |     |     |     |
| 430 |      |        |              |     |            |       |             | tionsoftraditionalgradient-basedmethodsthatstrictlyrely |     |     |     |     |     |
(Zhouetal.,2023),scaling(Kalogeropoulosetal.,2024))
| 431 |      |       |       |                |     |                |          | ondataaccess. |     |     |     |     |     |
| --- | ---- | ----- | ----- | -------------- | --- | -------------- | -------- | ------------- | --- | --- | --- | --- | --- |
|     | that | exist | in NN | model weights. |     | In our current | approach |               |     |     |     |     |     |
432
noeffortismadetoeitherrepresentthemodelsinaform Furtherresearchisrequiredtodevelopthisapproachinto
433 thatismoresymmetryinvariant,ortouseametanetwork a mature framework. Current limitations regarding inter-
434
architecturethatissymmetryinvariant. Thismeansthattoa modelvarianceandinter-classbiashighlighttheneedfor
435
largeextentdealingwiththesesymmetriesisdelegatedto morerobustexperimentation. Futureworkshouldinvesti-
436
themetanetwork. Relievingsomeofthisrecombinatorial gatelargermodelzoosandalternativeinputrepresentations
437
pressuremighthelpthemetanetworkachievebetterperfor- toimprovethegeneralizationandstabilityofweight-space
| 438 | mancewithfewerparameters. |     |     |     |     |     |     | modelediting. |     |     |     |     |     |
| --- | ------------------------- | --- | --- | --- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- |
439
8

UtilizingWeightSpaceLearningforData-FreeModelEditing
440 References Kalogeropoulos,I.,Bouritsas,G.,andPanagakis,Y. Scale
| 441 |                                           |     |     |     |     |              |       | EquivariantGraphMetanetworks.                    |     |     | InAdvancesinNeural |      |           |
| --- | ----------------------------------------- | --- | --- | --- | --- | ------------ | ----- | ------------------------------------------------ | --- | --- | ------------------ | ---- | --------- |
|     | Ainsworth,S.K.,Hayase,J.,andSrinivasa,S.  |     |     |     |     | Gitre-basin: |       |                                                  |     |     |                    |      |           |
| 442 |                                           |     |     |     |     |              |       | InformationProcessingSystems,volume37,pp.106800– |     |     |                    |      |           |
|     | Mergingmodelsmodulopermutationsymmetries. |     |     |     |     |              | arXiv |                                                  |     |     |                    |      |           |
| 443 |                                           |     |     |     |     |              |       | 106840.CurranAssociates,Inc.,2024.               |     |     |                    | doi: | 10.52202/ |
preprintarXiv:2209.04836,2022.
| 444 |        |              |     |                  |     |     |           | 079017-3391. |            |                        |     |     |          |
| --- | ------ | ------------ | --- | ---------------- | --- | --- | --------- | ------------ | ---------- | ---------------------- | --- | --- | -------- |
| 445 | Braun, | D., Bushnaq, | L., | Heimersheim,     |     | S., | Mendel,   |              |            |                        |     |     |          |
| 446 |        |              |     |                  |     |     |           | Knyazev,     | B., Hwang, | D., andLacoste-Julien, |     |     | S. CanWe |
|     | J.,    | and Sharkey, | L.  | Interpretability |     | in  | Parameter |              |            |                        |     |     |          |
ScaleTransformerstoPredictParametersofDiverseIm-
| 447 | Space: | Minimizing | Mechanistic |     | Description |     | Length |     |     |             |     |          |          |
| --- | ------ | ---------- | ----------- | --- | ----------- | --- | ------ | --- | --- | ----------- | --- | -------- | -------- |
|     |        |            |             |     |             |     |        |     |     | Proceedings | of  | the 40th | Interna- |
448 withAttribution-basedParameterDecomposition. arXiv ageNet Models? In
| 449 |          |                   |     |          |     |       |          | tionalConferenceonMachineLearning,volume202,pp. |     |     |     |     |     |
| --- | -------- | ----------------- | --- | -------- | --- | ----- | -------- | ----------------------------------------------- | --- | --- | --- | --- | --- |
|     | preprint | arXiv:2501.14926, |     | February |     | 2025. | doi: 10. |                                                 |     |     |     |     |     |
17243–17259.PMLR,23–29Jul2023.
450 48550/arXiv.2501.14926.
451
452 Bushnaq,L.,Braun,D.,andSharkey,L. StochasticParam- Krizhevsky, A., Hinton, G., et al. Learning mul-
|     |                    |     |                                |     |     |     |     | tiple layers | of features |     | from tiny | images. | 2009. |
| --- | ------------------ | --- | ------------------------------ | --- | --- | --- | --- | ------------ | ----------- | --- | --------- | ------- | ----- |
| 453 | eterDecomposition. |     | arXivpreprintarXiv:2506.20790, |     |     |     |     |              |             |     |           |         |       |
URL https://www.cs.toronto.edu/˜kriz/
| 454 | 2025. | doi: 10.48550/arXiv.2506.20790. |     |     |     |     |     |     |     |     |     |     |     |
| --- | ----- | ------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
learning-features-2009-TR.pdf.
455
Chundawat,V.S.,Tarun,A.K.,Mandal,M.,andKankan-
456
halli, M. Zero-shot machine unlearning. IEEE Trans- Lecun,Y.,Bottou,L.,Bengio,Y.,andHaffner,P. Gradient-
457
actionsonInformationForensicsandSecurity,18:2345– based learning applied to document recognition. Pro-
458
2354,2023. ceedings of the IEEE, 86(11):2278–2324, 1998. doi:
459
10.1109/5.726791.
| 460 | Eilertsen, | G., | Jo¨nsson, D., | Ropinski, | T., | Unger, | J., and |     |     |     |     |     |     |
| --- | ---------- | --- | ------------- | --------- | --- | ------ | ------- | --- | --- | --- | --- | --- | --- |
461
Ynnerman,A. Classifyingtheclassifier: dissectingthe Liang, Z., Tang, D., Zhou, Y., Zhao, X., Shi, M., Zhao,
462
weightspaceofneuralnetworks.InEuropeanConference W.,Li,Z.,Wang,P.,Schu¨rholt,K.,Borth,D.,Bronstein,
463 onArtificialIntelligence(ECAI2020),volume325,pp. M.M.,You,Y.,Wang,Z.,andWang,K. Drag-and-Drop
464
1119–1126.IOSPRESS,2020. LLMs: Zero-Shot Prompt-to-Weights. arXiv preprint
| 465 |     |     |     |     |     |     |     | arXiv:2506.16406,June2025. |     |     | doi: 10.48550/arXiv.2506. |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------------- | --- | --- | ------------------------- | --- | --- |
466 Falk,D.,Meynent,L.,Pfammatter,F.,Schu¨rholt,K.,and
16406.
| 467 | Borth, | D.  | A model zoo | of vision | transformers, |     | 2025. |     |     |     |     |     |     |
| --- | ------ | --- | ----------- | --------- | ------------- | --- | ----- | --- | --- | --- | --- | --- | --- |
URLhttps://arxiv.org/abs/2504.10231.
| 468 |     |     |     |     |     |     |     | Lim,D.,Maron,H.,andLaw,M.T.              |     |     | GraphMetanetworks |     |            |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------------------------------- | --- | --- | ----------------- | --- | ---------- |
| 469 |     |     |     |     |     |     |     | forProcessingDiverseNeuralArchitectures. |     |     |                   |     | ICLR,2024. |
Foster,J.,Fogarty,K.,Schoepf,S.,Dugue,Z.,O¨ztireli,C.,
470
|     | andBrintrup,A.     |     | Aninformationtheoreticapproachto |     |     |     |     |                                                   |            |         |      |           |      |
| --- | ------------------ | --- | -------------------------------- | --- | --- | --- | --- | ------------------------------------------------- | ---------- | ------- | ---- | --------- | ---- |
| 471 |                    |     |                                  |     |     |     |     | Mishra,A.,Kumar,T.,Nayak,G.,Shah,A.,Bhattacharya, |            |         |      |           |      |
|     | machineunlearning. |     | arXivpreprintarXiv:2402.01401,   |     |     |     |     |                                                   |            |         |      |           |      |
| 472 |                    |     |                                  |     |     |     |     | S., and                                           | Foltin, M. | Erasing | clip | memories: | Non- |
2024a.
| 473 |            |              |                    |           |     |           |         | destructive,data-freezero-shotclassunlearninginclip |           |                        |     |     |     |
| --- | ---------- | ------------ | ------------------ | --------- | --- | --------- | ------- | --------------------------------------------------- | --------- | ---------------------- | --- | --- | --- |
| 474 |            |              |                    |           |     |           |         | models,                                             | 2025. URL | https://arxiv.org/abs/ |     |     |     |
|     | Foster,    | J., Schoepf, | S., and            | Brintrup, | A.  | Fast      | machine |                                                     |           |                        |     |     |     |
| 475 |            |              |                    |           |     |           |         | 2512.14137.                                         |           |                        |     |     |     |
|     | unlearning |              | without retraining | through   |     | selective | synap-  |                                                     |           |                        |     |     |     |
476
|     | ticdampening. |     | InProceedingsoftheAAAIConference |     |     |     |     |                                                       |     |     |     |     |     |
| --- | ------------- | --- | -------------------------------- | --- | --- | --- | --- | ----------------------------------------------------- | --- | --- | --- | --- | --- |
| 477 |               |     |                                  |     |     |     |     | Moulin,O.,Francois-lavet,V.,Elbers,P.,andHoogendoorn, |     |     |     |     |     |
onArtificialIntelligence,volume38,pp.12043–12051,
M. Leveragingweightssignals–predictingandimprov-
| 478 | 2024b. | doi: | 10.1609/aaai.v38i11.29092. |     |     |     |     |     |     |     |     |     |     |
| --- | ------ | ---- | -------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
inggeneralizabilityinreinforcementlearning,2025.URL
479
480 Golatkar,A.,Achille,A.,andSoatto,S. Eternalsunshineof https://arxiv.org/abs/2511.20234.
| 481 | thespotlessnet: |     | Selectiveforgettingindeepnetworks. |     |     |     | In  |     |     |     |     |     |     |
| --- | --------------- | --- | ---------------------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Navon,A.,Shamsian,A.,Achituve,I.,Fetaya,E.,Chechik,
482 IEEE/CVFConferenceonComputerVisionandPattern
|     |     |     |     |     |     |     |     | G.,andMaron,H. |     | Equivariantarchitecturesforlearning |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------- | --- | ----------------------------------- | --- | --- | --- |
483 Recognition(CVPR),pp.9304–9312,2020.doi:10.1109/
|     |     |     |     |     |     |     |     | indeepweightspaces. |     | InProceedingsofthe40thInter- |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------- | --- | ---------------------------- | --- | --- | --- |
484 CVPR42600.2020.00932.
nationalConferenceonMachineLearning,volume202,
485
Ilharco,G.,Ribeiro,M.T.,Wortsman,M.,Gururangan,S., pp.25790–25816.PMLR,23–29Jul2023.
486
|     | Schmidt,L.,Hajishirzi,H.,andFarhadi,A. |     |     |     |     | Editingmod- |     |     |     |     |     |     |     |
| --- | -------------------------------------- | --- | --- | --- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- |
487
|     |     |     |     |     |     |     |     | Netzer, Y., | Wang, T., | Coates, | A., Bissacco, |     | A., Wu, B., |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | --------- | ------- | ------------- | --- | ----------- |
elswithtaskarithmetic.arXivpreprintarXiv:2212.04089,
| 488 |     |     |     |     |     |     |     | Ng, A. | Y., et al. | Reading | digits | in natural | images |
| --- | --- | --- | --- | --- | --- | --- | --- | ------ | ---------- | ------- | ------ | ---------- | ------ |
2022a.
489
|     |     |     |     |     |     |     |     | withunsupervisedfeaturelearning. |     |     |     | InNIPSworkshop |     |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------------------------- | --- | --- | --- | -------------- | --- |
490
Ilharco,G.,Wortsman,M.,Gadre,S.Y.,Song,S.,Hajishirzi, on deep learning and unsupervised feature learn-
491
H.,Kornblith,S.,Farhadi,A.,andSchmidt,L. Patching ing, volume 2011, pp. 7, 2011. URL https:
492
open-vocabularymodelsbyinterpolatingweights. vol- //www-cs.stanford.edu/˜twangcat/
493 ume35,pp.29262–29277,2022b. papers/nips2011_housenumbers.pdf.
494
9

UtilizingWeightSpaceLearningforData-FreeModelEditing
495 Nguyen,T.T.,Huynh,T.T.,Ren,Z.,Nguyen,P.L.,Liew, Vaswani,A.,Shazeer,N.,Parmar,N.,Uszkoreit,J.,Jones,
496 A. W.-C., Yin, H., and Nguyen, Q. V. H. A Survey of L.,Gomez,A.N.,Kaiser,L.u.,andPolosukhin,I. Atten-
497 MachineUnlearning. ACMTransactionsonIntellingent tionisAllyouNeed. InAdvancesinNeuralInformation
498 SystemsandTechnology,16(5),2025. ISSN2157-6904. ProcessingSystems,volume30.CurranAssociates,Inc.,
| 499 | doi: 10.1145/3749987. |     |     | 2017. |     |     |     |
| --- | --------------------- | --- | --- | ----- | --- | --- | --- |
500
Radford,A.,Kim,J.W.,Hallacy,C.,Ramesh,A.,Goh,G., Wang, K., Tang, D., Zhao, W., Schu¨rholt, K., Wang, Z.,
501
Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, and You, Y. Recurrent Diffusion for Large-Scale Pa-
502
J., Krueger, G., and Sutskever, I. Learning Transfer- rameter Generation. Advances in Neural Information
503
504 ableVisualModelsfromNaturalLanguageSupervision. Processing Systems, December 2025. doi: 10.48550/
InProceedingsofthe38thInternationalConferenceon arXiv.2501.11587. URLhttp://arxiv.org/abs/
505
|     | MachineLearning,volume139,pp.8748–8763.PMLR, |     |     | 2501.11587. |     |     |     |
| --- | -------------------------------------------- | --- | --- | ----------- | --- | --- | --- |
506
507 18–24Jul2021.
|     |     |     |     | Xiao, H., Rasul, | K., and Vollgraf, | R. Fashion-mnist: | a   |
| --- | --- | --- | --- | ---------------- | ----------------- | ----------------- | --- |
508
Rangel,J.M.L.,Schoepf,S.,Foster,J.,Krueger,D.,and novelimagedatasetforbenchmarkingmachinelearning
509
Anwar,U. Learningtoforgetusinghypernetworks. arXiv algorithms. arXivpreprintarXiv:1708.07747,2017. doi:
510
preprintarXiv:2412.00761,2024. doi: 10.48550/arXiv. 10.48550/arXiv.1708.07747.
511
2412.00761.
| 512 |     |     |     | Zhou, A., Yang, | K., Burns, | K., Cardace, | A., Jiang, Y., |
| --- | --- | --- | --- | --------------- | ---------- | ------------ | -------------- |
513 Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Sokota, S., Kolter, J. Z., and Finn, C. Permutation
| 514 |                     |                     |             | EquivariantNeuralFunctionals. |     | AdvancesinNeuralIn- |     |
| --- | ------------------- | ------------------- | ----------- | ----------------------------- | --- | ------------------- | --- |
|     | Klimov, O. Proximal | policy optimization | algorithms. |                               |     |                     |     |
515 arXivpreprintarXiv:1707.06347,2017. doi: 10.48550/ formation Processing Systems, September 2023. doi:
516 arXiv.1707.06347. 10.48550/arXiv.2302.14040. URL http://arxiv.
| 517 |     |     |     | org/abs/2302.14040. |     |     |     |
| --- | --- | --- | --- | ------------------- | --- | --- | --- |
Schu¨rholt,K.,Mahoney,M.W.,andBorth,D.Towardsscal-
518
| 519 | ableandversatileweightspacelearning. |     | InProceedings |     |     |     |     |
| --- | ------------------------------------ | --- | ------------- | --- | --- | --- | --- |
ofthe41stInternationalConferenceonMachineLearn-
520
https://proceedings.
| 521 | ing. JMLR, 2024. | URL |     |     |     |     |     |
| --- | ---------------- | --- | --- | --- | --- | --- | --- |
522 mlr.press/v235/schurholt24a.html.
523
| Schu¨rholt, | K., Kostadinov, | D., and Borth, | D. Self- |     |     |     |     |
| ----------- | --------------- | -------------- | -------- | --- | --- | --- | --- |
524
SupervisedRepresentationLearningonNeuralNetwork
525
WeightsforModelCharacteristicPrediction.InAdvances
526
|     | in Neural Information | Processing Systems, | volume 34, |     |     |     |     |
| --- | --------------------- | ------------------- | ---------- | --- | --- | --- | --- |
527
pp.16481–16493.CurranAssociates,Inc.,2021.
528
529
Schu¨rholt,K.,Knyazev,B.,Giro´-iNieto,X.,andBorth,D.
530
|     | Hyper-RepresentationsasGenerativeModels: |                  | Sampling |     |     |     |     |
| --- | ---------------------------------------- | ---------------- | -------- | --- | --- | --- | --- |
| 531 |                                          | AdvancesinNeural |          |     |     |     |     |
UnseenNeuralNetworkWeights.
532
InformationProcessingSystems,35:27906–27920,De-
533 cember2022a.
534
535 Schu¨rholt,K.,Taskiran,D.,Knyazev,B.,Giro´-iNieto,X.,
| 536 | andBorth,D. ModelZoos:        | ADatasetofDiversePopu- |     |     |     |     |     |
| --- | ----------------------------- | ---------------------- | --- | --- | --- | --- | --- |
| 537 | lationsofNeuralNetworkModels. | AdvancesinNeural       |     |     |     |     |     |
538 InformationProcessingSystems,35:38134–38148,De-
539 cember2022b.
540
Tarun,A.K.,Chundawat,V.S.,Mandal,M.,andKankan-
541
|     | halli, M. Fastyeteffectivemachineunlearning. |     | IEEE |     |     |     |     |
| --- | -------------------------------------------- | --- | ---- | --- | --- | --- | --- |
542
TransactionsonNeuralNetworksandLearningSystems,
543
35(9):13046–13055,2023.
544
545
Unterthiner,T.,Keysers,D.,Gelly,S.,Bousquet,O.,and
546
|     | Tolstikhin,I. PredictingNeuralNetworkAccuracyfrom |     |     |     |     |     |     |
| --- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
547
|     | Weights. arXivpreprintarXiv:2002.11448,April2021. |     |     |     |     |     |     |
| --- | ------------------------------------------------- | --- | --- | --- | --- | --- | --- |
| 548 | doi: 10.48550/arXiv.2002.11448.                   |     |     |     |     |     |     |
549
10

UtilizingWeightSpaceLearningforData-FreeModelEditing
550 A.MoreRelatedwork
551
A.1.Parameterdecomposition
552
553 IntheirworkonAttribution-basedParameterDecomposition(Braunetal.,2025)positthatwhentraininganeuralnetwork,
554 theoptimizationalgorithmiscompressingtheinformationthatisrelevanttothetaskintoanetworksweights. Forexample,
555 awell-trainedLLMwillprobablycontaintheknowledgethat”theskyisblue”. Thisshouldbestoredsomewhereinits
556 weights. Thenaturalcentralquestionisthen: Canthisinformationbelocatedandseparatedfromallotherknowledge? Is
557 thereasubnetwork,acomponentoftheparameters,whichactivateswheneverthisknowledgemustbeusedinaforward
558 pass,butneverotherwise? Theauthorsintroduceanoveltechniquethatshowspromisingsignsofdoingexactlythis.
559
ThecorehypothesisofBraunetal.(2025)isthatiftheweightspacecontainsatomizedmechanisms(APD/SPD),then
560
evidenceofknowledgeshouldberetrievablefromthemodels’weights. Thereisaninformation-theoreticalquestionthat
561
underlies this hypothesis: can the information that the model has learned be explained in full by its weights? Or does
562
additionalcriticalinformationonlyexistincombinationwiththedataorarchitecture?
563
564 Ourworkintersectswiththisparadigm. Ratherthandecomposingsinglenetworks,weaskwhethercommonmechanismsor
565 weight-pattern“signatures”emergeacrossapopulationofnetworks.Forinstance,ifmanyCNNslearnthesametask,dothey
566 convergetosimilarweightcomponentsorcircuits? Wewereinspiredbythehypothesisthatifweight-spacemechanismsare
567 real,evidenceofthemshouldbedetectablebyametanetworkscanningacrossmanynetworks
568
569
A.2.RecentdevelopmentsofWSL
570
571 A.2.1.SCALINGANDVERSATILITY
572
Inrecentyearsimpressivestepshavebeenmadetoaddresssymmetries,scaling,andgeneralizationtodiversefeedforward
573
architecturesinWSL.
574
575 Zhouetal.(2023)introduceanovelarchitecture: NeuralFunctionNetworks(NFNs)toaddresspermutationsymmetries.
576 Limetal.(2024)presentedamethodofrepresentinganNNasaparametergraph,whichhasthebenefitofbeingparameter
577 permutationsymmetry-equivariant,butunlikeestablishedcomputationgraphrepresentationsallowforefficienthandlingof
578 parameter-sharinglayers. Theseparametergraphsaremodularandflexible,andGraphNeuralNetworkscanthenbetrained
579 forarangeofdownstreamtasksontheparametergraphs,suchaspredictionandparametergeneration. Kalogeropoulos
580 etal.(2024)extendthismethodologybyalsoaddressingscalesymmetriesthatresultfromusingReLUactivationfunctions.
581 Schu¨rholtetal.(2024)utilizesequentialtransformerstolearntask-agnosticrepresentationsofmodels. Duetothesequential
582 processing of neural network weights this method is very scalable. These developments suggest that, if an effective
583 model-steeringmethodcanbeestablished,thereisreasontoexpectitwillbescalable.
584
585 A.2.2.SELF-SUPERVISEDWSL
586
Another approach is to use self-supervised learning (SSL) where metanetworks create hyper-representations of model
587
weights. Theselearnedrepresentationscanbeusedtopredicthyperparameters,testaccuracy,andgeneralization(Schu¨rholt
588
etal.,2021),butalsoasabasisforthegenerationofunseenNNs(Schu¨rholtetal.,2022a).
589
590 In a similar vein, multiple other methods are proposed which train a neural network to generate new instances of NN
591 parameters(Knyazevetal.,2023;Wangetal.,2025;Liangetal.,2025),inthiscontextreferredtoasahypernetworks.
592 Knyazevetal.(2023)haveutilizedTransformermodels(Vaswanietal.,2017). Forthegenerationoffavorableinitializations
593 ofImageNetModels. Wangetal.(2025)andLiangetal.(2025)havescaledtheuseofhypernetworks(metanetworks)to
594 generatefullNNparametersforsizesofupto100million,fromLLMstoResNetsandvisiontransformers(ViTs).
595
596
597
598
599
600
601
602
603
604
11

UtilizingWeightSpaceLearningforData-FreeModelEditing
605 B.Unlearningresultsperclass
606
B.1.MNIST
607
608
609
Table4.Unlearning:MNISTper-classresults(classes0–4).Samplesizes:0(814),1(70),2(4711),3(2922),4(4160).
610
| 611 |        | 0     | 1       | 2       | 3       | 4         |
| --- | ------ | ----- | ------- | ------- | ------- | --------- |
|     |        | D ↓ D | ↑ D ↓ D | ↑ D ↓ D | ↑ D ↓ D | ↑ D ↓ D ↑ |
| 612 | Method | t     | r t     | r t     | r t     | r t r     |
613
|     | Before | 0.86 0.77 | 0.97 0.42 | 0.85 0.90 | 0.85 0.85 | 0.91 0.90 |
| --- | ------ | --------- | --------- | --------- | --------- | --------- |
614
|     | Retainfinetune | 0.00 0.78 | 0.82 0.42 | 0.06 0.90 | 0.00 0.86 | 0.00 0.91 |
| --- | -------------- | --------- | --------- | --------- | --------- | --------- |
615
|     | Gradientascent | 0.00 0.21 | 0.00 0.17 | 0.00 0.13 | 0.00 0.11 | 0.00 0.11 |
| --- | -------------- | --------- | --------- | --------- | --------- | --------- |
616
|     | Randomvector | 0.77 0.64 | 0.95 0.36 | 0.77 0.81 | 0.83 0.77 | 0.88 0.86 |
| --- | ------------ | --------- | --------- | --------- | --------- | --------- |
617
|     | Ourmethod | 0.69 0.54 | 0.63 0.36 | 0.41 0.69 | 0.85 0.66 | 0.74 0.77 |
| --- | --------- | --------- | --------- | --------- | --------- | --------- |
618
619
620
621
622
Table5.Unlearning:MNISTper-classresults(classes5–9).Samplesizes:5(4350),6(715),7(1269),8(3484),9(4526).
623
| 624 |     | 5   | 6   | 7   | 8   | 9   |
| --- | --- | --- | --- | --- | --- | --- |
625 Method D t ↓ D r ↑ D t ↓ D r ↑ D t ↓ D r ↑ D t ↓ D r ↑ D t ↓ D r ↑
626
|     | Before | 0.86 0.92 | 0.78 0.63 | 0.87 0.82 | 0.87 0.88 | 0.88 0.90 |
| --- | ------ | --------- | --------- | --------- | --------- | --------- |
627
|     | Retainfinetune | 0.01 0.93 | 0.05 0.65 | 0.00 0.83 | 0.01 0.89 | 0.00 0.92 |
| --- | -------------- | --------- | --------- | --------- | --------- | --------- |
628
|     | Gradientascent | 0.00 0.17 | 0.00 0.31 | 0.00 0.11 | 0.00 0.23 | 0.00 0.14 |
| --- | -------------- | --------- | --------- | --------- | --------- | --------- |
629
|     | Randomvector | 0.78 0.88 | 0.73 0.59 | 0.84 0.71 | 0.84 0.82 | 0.82 0.84 |
| --- | ------------ | --------- | --------- | --------- | --------- | --------- |
| 630 | Ourmethod    |           |           |           |           |           |
|     |              | 0.54 0.81 | 0.72 0.50 | 0.73 0.63 | 0.75 0.77 | 0.38 0.76 |
631
632
633
634 B.2.Fashion-MNIST
635
636
637 Table6.Unlearning:Fashion-MNISTper-classresults(classes0–4).Samplesizes:0(3409),1(1244),2(4539),3(2436),4(4009).
638
|     |     | 0   | 1   | 2   | 3   | 4   |
| --- | --- | --- | --- | --- | --- | --- |
639
|     | Method | D ↓ D     | ↑ D ↓ D   | ↑ D ↓ D   | ↑ D ↓ D   | ↑ D ↓ D ↑ |
| --- | ------ | --------- | --------- | --------- | --------- | --------- |
| 640 |        | t         | r t       | r t       | r t       | r t r     |
| 641 | Before | 0.75 0.75 | 0.89 0.65 | 0.73 0.78 | 0.79 0.73 | 0.71 0.79 |
642 Retainfinetune 0.00 0.77 0.08 0.66 0.00 0.81 0.01 0.75 0.00 0.83
|     | Gradientascent | 0.00 0.21 | 0.00 0.13 | 0.00 0.25 | 0.00 0.11 | 0.00 0.12 |
| --- | -------------- | --------- | --------- | --------- | --------- | --------- |
643
644 Randomvector 0.72 0.69 0.89 0.61 0.68 0.75 0.76 0.70 0.69 0.77
645 Ourmethod 0.49 0.64 0.85 0.54 0.43 0.71 0.63 0.62 0.08 0.64
646
647
648
649
650 Table7.Fashion-MNISTper-classresults(classes5–9).Samplesizes:5(5225),6(4118),7(2911),8(1774),9(2421).
651
|     |     | 5   | 6   | 7   | 8   | 9   |
| --- | --- | --- | --- | --- | --- | --- |
652
|     | Method | D ↓ D | ↑ D ↓ D | ↑ D ↓ D | ↑ D ↓ D | ↑ D ↓ D ↑ |
| --- | ------ | ----- | ------- | ------- | ------- | --------- |
|     |        | t     | r t     | r t     | r t     | r t r     |
653
| 654 | Before | 0.83 0.74 | 0.26 0.84 | 0.86 0.73 | 0.87 0.68 | 0.94 0.70 |
| --- | ------ | --------- | --------- | --------- | --------- | --------- |
655 Retainfinetune 0.02 0.74 0.00 0.85 0.00 0.75 0.07 0.69 0.02 0.71
|     | Gradientascent | 0.00 0.12 | 0.00 0.30 | 0.00 0.13 | 0.00 0.19 | 0.00 0.11 |
| --- | -------------- | --------- | --------- | --------- | --------- | --------- |
656
657 Randomvector 0.79 0.69 0.30 0.81 0.85 0.71 0.85 0.64 0.92 0.65
|     | Ourmethod | 0.13 0.62 | 0.12 0.80 | 0.56 0.68 | 0.85 0.58 | 0.68 0.54 |
| --- | --------- | --------- | --------- | --------- | --------- | --------- |
658
659
12

UtilizingWeightSpaceLearningforData-FreeModelEditing
660 B.3.CIFAR-10
661
662
Table8.CIFAR-10per-classresults(classes0–4).Samplesizes:0(3193),1(2394),2(3346),3(1284),4(2644).
663
664
|     |     | 0   | 1   | 2   | 3   | 4   |
| --- | --- | --- | --- | --- | --- | --- |
665
|     | Method | D ↓ D     | ↑ D ↓ D   | ↑ D ↓ D   | ↑ D ↓ D   | ↑ D ↓ D ↑ |
| --- | ------ | --------- | --------- | --------- | --------- | --------- |
| 666 |        | t         | r t       | r t       | r t       | r t r     |
| 667 | Before | 0.39 0.37 | 0.50 0.37 | 0.22 0.39 | 0.16 0.44 | 0.35 0.39 |
668 Retainfinetune 0.00 0.39 0.00 0.40 0.00 0.40 0.00 0.45 0.00 0.41
669 Gradientascent 0.00 0.12 0.00 0.12 0.00 0.11 0.00 0.12 0.00 0.13
670 Randomvector 0.33 0.32 0.47 0.33 0.15 0.33 0.14 0.38 0.28 0.34
671 Ourmethod 0.03 0.34 0.15 0.35 0.05 0.34 0.02 0.38 0.13 0.35
672
673
674
675
676 Table9.CIFAR-10per-classresults(classes5–9).Samplesizes:5(2905),6(2662),7(2112),8(3374),9(3038).
677
| 678 |        | 5     | 6       | 7       | 8       | 9         |
| --- | ------ | ----- | ------- | ------- | ------- | --------- |
| 679 | Method | D ↓ D | ↑ D ↓ D | ↑ D ↓ D | ↑ D ↓ D | ↑ D ↓ D ↑ |
|     |        | t     | r t     | r t     | r t     | r t r     |
680
|     | Before | 0.47 0.37 | 0.54 0.37 | 0.38 0.40 | 0.55 0.35 | 0.45 0.36 |
| --- | ------ | --------- | --------- | --------- | --------- | --------- |
681
|     | Retainfinetune | 0.00 0.40 | 0.00 0.40 | 0.00 0.41 | 0.00 0.39 | 0.00 0.39 |
| --- | -------------- | --------- | --------- | --------- | --------- | --------- |
682
|     | Gradientascent | 0.00 0.11 | 0.00 0.13 | 0.00 0.11 | 0.00 0.11 | 0.00 0.12 |
| --- | -------------- | --------- | --------- | --------- | --------- | --------- |
683
|     | Randomvector | 0.39 0.33 | 0.49 0.33 | 0.32 0.36 | 0.47 0.31 | 0.41 0.32 |
| --- | ------------ | --------- | --------- | --------- | --------- | --------- |
684
|     | Ourmethod | 0.22 0.34 | 0.32 0.33 | 0.18 0.36 | 0.15 0.33 | 0.23 0.33 |
| --- | --------- | --------- | --------- | --------- | --------- | --------- |
685
686
687
688
B.4.SVHNcropped
689
690
691
Table10.Unlearning:SVHNper-classresults(classes0–4).Samplesizes:0(2166),1(709),2(1911),3(1870),4(1283).
692
| 693 |        | 0     | 1       | 2       | 3       | 4         |
| --- | ------ | ----- | ------- | ------- | ------- | --------- |
| 694 | Method | D ↓ D | ↑ D ↓ D | ↑ D ↓ D | ↑ D ↓ D | ↑ D ↓ D ↑ |
|     |        | t     | r t     | r t     | r t     | r t r     |
695
|     | Before | 0.37 0.42 | 0.79 0.28 | 0.79 0.33 | 0.42 0.43 | 0.51 0.44 |
| --- | ------ | --------- | --------- | --------- | --------- | --------- |
696
|     | Retainfinetune | 0.00 0.42 | 0.00 0.32 | 0.00 0.40 | 0.00 0.45 | 0.00 0.46 |
| --- | -------------- | --------- | --------- | --------- | --------- | --------- |
697
|     | Gradientascent | 0.00 0.12 | 0.00 0.11 | 0.00 0.11 | 0.00 0.11 | 0.00 0.11 |
| --- | -------------- | --------- | --------- | --------- | --------- | --------- |
698
|     | Randomvector | 0.26 0.34 | 0.58 0.21 | 0.60 0.23 | 0.32 0.35 | 0.42 0.37 |
| --- | ------------ | --------- | --------- | --------- | --------- | --------- |
699
|     | Ourmethod | 0.27 0.33 | 0.37 0.21 | 0.43 0.24 | 0.22 0.35 | 0.28 0.38 |
| --- | --------- | --------- | --------- | --------- | --------- | --------- |
700
701
702
703
704
705 Table11.Unlearning:SVHNper-classresults(classes5–9).Samplesizes:5(1793),6(823),7(1517),8(1289),9(477).
706
|     |     | 5   | 6   | 7   | 8   | 9   |
| --- | --- | --- | --- | --- | --- | --- |
707
|     | Method | D ↓ D | ↑ D ↓ D | ↑ D ↓ D | ↑ D ↓ D | ↑ D ↓ D ↑ |
| --- | ------ | ----- | ------- | ------- | ------- | --------- |
|     |        | t     | r t     | r t     | r t     | r t r     |
708
| 709 | Before | 0.56 0.41 | 0.28 0.52 | 0.55 0.42 | 0.27 0.49 | 0.31 0.61 |
| --- | ------ | --------- | --------- | --------- | --------- | --------- |
710 Retainfinetune 0.00 0.45 0.00 0.54 0.00 0.42 0.00 0.49 0.00 0.62
|     | Gradientascent | 0.00 0.11 | 0.00 0.11 | 0.00 0.12 | 0.00 0.12 | 0.00 0.12 |
| --- | -------------- | --------- | --------- | --------- | --------- | --------- |
711
712 Randomvector 0.48 0.36 0.24 0.46 0.42 0.37 0.19 0.40 0.24 0.51
|     | Ourmethod | 0.51 0.35 | 0.19 0.46 | 0.35 0.37 | 0.17 0.40 | 0.16 0.53 |
| --- | --------- | --------- | --------- | --------- | --------- | --------- |
713
714
13

UtilizingWeightSpaceLearningforData-FreeModelEditing
Table16.Uplearning:CIFAR-10per-classresults(classes0–4).Samplesizes:0(722),1(1056),2(983),3(904),4(1059).
715
716
|     |     | 0   | 1   | 2   | 3   | 4   |
| --- | --- | --- | --- | --- | --- | --- |
717
|     | Method | D ↑ D     | D ↑ D     | D ↑ D     | D ↑ D     | D ↑ D     |
| --- | ------ | --------- | --------- | --------- | --------- | --------- |
| 718 |        | t         | r t       | r t       | r t       | r t r     |
| 719 | Before | 0.38 0.36 | 0.42 0.33 | 0.18 0.36 | 0.06 0.36 | 0.33 0.34 |
720 Randomvector 0.23 0.27 0.29 0.24 0.01 0.20 0.02 0.23 0.09 0.24
721 Ourmethod 0.59 0.22 0.38 0.22 0.36 0.15 0.53 0.16 0.35 0.20
722
723
724
C.Performanceimprovementresultsperclass
725
726
C.1.MNIST
727
728
Table12.Uplearning:MNISTper-classresults(classes0–4).Samplesizes:0(137),1(4),2(383),3(315),4(262).
729
| 730 |     | 0   | 1   | 2   | 3   | 4   |
| --- | --- | --- | --- | --- | --- | --- |
731
|     | Method | D ↑ D     | D ↑ D     | D ↑ D     | D ↑ D     | D ↑ D     |
| --- | ------ | --------- | --------- | --------- | --------- | --------- |
| 732 |        | t         | r t       | r t       | r t       | r t r     |
|     | Before | 0.50 0.51 | 0.77 0.17 | 0.61 0.76 | 0.78 0.71 | 0.51 0.69 |
733
734 Randomvector 0.42 0.48 0.00 0.11 0.56 0.69 0.76 0.63 0.45 0.63
|     | Ourmethod | 0.41 0.52 | 1.00 0.00 | 0.69 0.66 | 0.73 0.62 | 0.75 0.57 |
| --- | --------- | --------- | --------- | --------- | --------- | --------- |
735
736
737
738
739 Table13.Uplearning:MNISTper-classresults(classes5–9).Samplesizes:5(430),6(119),7(292),8(334),9(356).
740
| 741 |        | 5         | 6         | 7         | 8         | 9         |
| --- | ------ | --------- | --------- | --------- | --------- | --------- |
| 742 | Method | D ↑ D     | D ↑ D     | D ↑ D     | D ↑ D     | D ↑ D     |
|     |        | t         | r t       | r t       | r t       | r t r     |
| 743 | Before | 0.46 0.81 | 0.59 0.48 | 0.85 0.72 | 0.71 0.72 | 0.74 0.74 |
744
|     | Randomvector | 0.38 0.74 | 0.61 0.45 | 0.84 0.65 | 0.68 0.66 | 0.71 0.67 |
| --- | ------------ | --------- | --------- | --------- | --------- | --------- |
745 Ourmethod 0.63 0.71 0.53 0.48 0.86 0.62 0.78 0.64 0.73 0.62
746
747
748
749 C.2.Fashion-MNIST
750
751 Table14.Uplearning:Fashion-MNISTper-classresults(classes0–4).Samplesizes:0(169),1(2),2(162),3(216),4(226).
752
| 753 |        | 0     | 1     | 2     | 3     | 4     |
| --- | ------ | ----- | ----- | ----- | ----- | ----- |
| 754 | Method | D ↑ D | D ↑ D | D ↑ D | D ↑ D | D ↑ D |
|     |        | t     | r t   | r t   | r t   | r t r |
755
|     | Before | 0.73 0.73 | 0.85 0.63 | 0.72 0.78 | 0.79 0.74 | 0.66 0.76 |
| --- | ------ | --------- | --------- | --------- | --------- | --------- |
756
|     | Randomvector | 0.69 0.63 | 0.82 0.62 | 0.66 0.73 | 0.75 0.68 | 0.60 0.71 |
| --- | ------------ | --------- | --------- | --------- | --------- | --------- |
757
|     | Ourmethod | 0.80 0.54 | 0.71 0.62 | 0.64 0.63 | 0.64 0.62 | 0.84 0.61 |
| --- | --------- | --------- | --------- | --------- | --------- | --------- |
758
759
760
761
Table15.Uplearning:Fashion-MNISTper-classresults(classes5–9).Samplesizes:5(76),6(216),7(115),8(55),9(2).
762
763
|     |     | 5   | 6   | 7   | 8   | 9   |
| --- | --- | --- | --- | --- | --- | --- |
764
|     | Method | D ↑ D     | D ↑ D     | D ↑ D     | D ↑ D     | D ↑ D     |
| --- | ------ | --------- | --------- | --------- | --------- | --------- |
| 765 |        | t         | r t       | r t       | r t       | r t r     |
|     | Before | 0.76 0.72 | 0.18 0.80 | 0.82 0.67 | 0.74 0.54 | 0.34 0.25 |
766
767 Randomvector 0.76 0.70 0.22 0.61 0.81 0.64 0.68 0.50 0.50 0.07
|     | Ourmethod | 0.87 0.65 | 0.91 0.31 | 0.84 0.57 | 0.71 0.54 | 0.34 0.16 |
| --- | --------- | --------- | --------- | --------- | --------- | --------- |
768
769
14

UtilizingWeightSpaceLearningforData-FreeModelEditing
770 Table17.Uplearning:CIFAR-10performanceimprovementper-classresults(classes5–9).Samplesizes:5(1069),6(1064),7(152),8
(1064),9(768).
771
772
5 6 7 8 9
773
Method D ↑ D D ↑ D D ↑ D D ↑ D D ↑ D
774 t r t r t r t r t r
775 Before 0.43 0.33 0.45 0.34 0.11 0.28 0.51 0.32 0.44 0.36
776 Randomvector 0.16 0.23 0.27 0.24 0.00 0.17 0.36 0.26 0.37 0.28
777 Ourmethod 0.84 0.15 0.59 0.19 0.65 0.10 0.78 0.19 0.65 0.24
778
779
780
781
782
783
784
785
786
787
788
789
790
791
792
793
794
795
796
797
798
799
800
801
802
803
804
805
806
807
808
809
810
811
812
813
814
815
816
817
818
819
820
821
822
823
824
15

UtilizingWeightSpaceLearningforData-FreeModelEditing
825 C.3.CIFAR-10
826
D.Unlearningresultexamples
827
828
829
MNIST
830
831
832
833
834
1.0
835
836
0.8
837
838
839 0.6
840
841 0.4
842
843
0.2
844
845
0.0
846
847 1.0
848
849 0.8
850
851
0.6
852
853
0.4
854
855
856 0.2
857
858 0.0
859 1.0
860
861
0.8
862
863
864 0.6
865
866 0.4
867
868
0.2
869
870
0.0
871 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
872
Classes
873
874
875
876
877
878
879
ycaruccA
Before intervention Predicted After Target (After)
After intervention Target (Before) Target (Prediction)
Figure6.Anunbiasedsampleof9MNISTmodelsunlearnedusingourmethod.Barplotshowsaccuraciesperclassonthetestsetbefore
intervention,afterinterventionandthepredictedaccuraciesbythemetanetworkafterintervention.Targetclassinorange.
16

UtilizingWeightSpaceLearningforData-FreeModelEditing
880
881
882
883
884
885
886
887
888
889
890 1.0
891
892 0.8
893
894 0.6
895
896
0.4
897
898
0.2
899
900
0.0
901
902 1.0
903
904 0.8
905
906 0.6
907
908 0.4
909
910
0.2
911
912
0.0
913
1.0
914
915
916 0.8
917
918 0.6
919
920 0.4
921
922
0.2
923
924
0.0
925 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
926 Classes
927
928
929
930
931
932
933
934
ycaruccA
Fashion-MNIST
Before intervention Predicted After Target (After)
After intervention Target (Before) Target (Prediction)
Figure7.Anunbiasedsampleof9FashionMNISTmodelsunlearnedusingourmethod.Barplotshowsaccuraciesperclassonthetestset
beforeintervention,afterinterventionandthepredictedaccuraciesbythemetanetworkafterintervention.Targetclassinorange.
17

UtilizingWeightSpaceLearningforData-FreeModelEditing
935
936
937
938
939 CIFAR-10
940
941
942
943
944
945 0.8
946 0.7
947
0.6
948
949 0.5
950 0.4
951
0.3
952
953 0.2
954 0.1
955
0.0
956
957
0.8
958
0.7
959
960 0.6
961
0.5
962
0.4
963
964 0.3
965 0.2
966
0.1
967
968 0.0
969
0.8
970
971 0.7
972
0.6
973
0.5
974
975 0.4
976
0.3
977
0.2
978
979 0.1
980
0.0
981 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
982
Classes
983
984
985
986
987
988
989
ycaruccA
Before intervention Predicted After Target (After)
After intervention Target (Before) Target (Prediction)
Figure8. Anunbiasedsampleof9CIFAR-10modelsunlearnedusingourmethod. Barplotshowsaccuraciesperclassonthetestset
beforeintervention,afterinterventionandthepredictedaccuraciesbythemetanetworkafterintervention.Targetclassinorange.
18

UtilizingWeightSpaceLearningforData-FreeModelEditing
990
991
992
993
994 SVHN
995
996
997
998
999
1000
0.8
1001
1002
1003 0.6
1004
1005
0.4
1006
1007
1008 0.2
1009
1010
0.0
1011
1012
1013 0.8
1014
1015
0.6
1016
1017
1018 0.4
1019
1020
0.2
1021
1022
1023 0.0
1024
1025
0.8
1026
1027
1028 0.6
1029
1030
0.4
1031
1032
1033 0.2
1034
1035
0.0
1036 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9
1037
Classes
1038
1039
1040
1041
1042
1043
1044
ycaruccA
Before intervention Predicted After Target (After)
After intervention Target (Before) Target (Prediction)
Figure9.Anunbiasedsampleof9SVHNmodelsunlearnedusingourmethod.Barplotshowsaccuraciesperclassonthetestsetbefore
intervention,afterinterventionandthepredictedaccuraciesbythemetanetworkafterintervention.Targetclassinorange.
19

UtilizingWeightSpaceLearningforData-FreeModelEditing
1045 E.UnfilteredData
1046
1047
1048
1049
1050
1051
1052
1053
1054
1055
1056
1057
1058
1059
1060
1061
1062
1063
1064
1065
1066
1067
1068
1069
1070
1071
1072
1073
1074
1075
1076
1077
1078
1079
1080
1081
1082
1083
1084
1085
1086
1087
1088
1089
1090
1091
1092
1093
1094
1095
1096
1097
1098
1099
20

UtilizingWeightSpaceLearningforData-FreeModelEditing
1100
1101
1102
1103
1104
1105
1106
1107
1108
|     |     | MNIST |     |     |     | Fashion-MNIST |     |     |
| --- | --- | ----- | --- | --- | --- | ------------- | --- | --- |
1109
|     | 1.00 |     |     |     | 1.00 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1110
| 1111 | 0.75 |     |     |     | 0.75 |     |     |     |
| ---- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1112
|      | 0.50              |     |     |     | 0.50              |     |     |     |
| ---- | ----------------- | --- | --- | --- | ----------------- | --- | --- | --- |
| 1113 | ecnereffiD tegraT |     |     |     | ecnereffiD tegraT |     |     |     |
1114
|     | 0.25 |     |     |     | 0.25 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1115
1116
|     | 0.00 |     |     |     | 0.00 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1117
| 1118 | 0.25 |     |     |     | 0.25 |     |     |     |
| ---- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1119
|     | 0.50 |     |     |     | 0.50 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1120
1121
|     | 0.75 |     |     |     | 0.75 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1122
1123
|     | 1.00 |     |     |     | 1.00 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1124
|      | 9 2 | 4 5 8        | 1 6 7 | 0 3 | 5 4 | 2 7 9        | 0 3 6 | 1 8 |
| ---- | --- | ------------ | ----- | --- | --- | ------------ | ----- | --- |
| 1125 |     | Target Class |       |     |     | Target Class |       |     |
1126
| 1127 |      | CIFAR-10 |     |     |      | SVHN |     |     |
| ---- | ---- | -------- | --- | --- | ---- | ---- | --- | --- |
|      | 1.00 |          |     |     | 1.00 |      |     |     |
1128
1129
|     | 0.75 |     |     |     | 0.75 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1130
| 1131 | 0.50              |     |     |     | 0.50              |     |     |     |
| ---- | ----------------- | --- | --- | --- | ----------------- | --- | --- | --- |
|      | ecnereffiD tegraT |     |     |     | ecnereffiD tegraT |     |     |     |
1132
|     | 0.25 |     |     |     | 0.25 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1133
1134
| 1135 | 0.00 |     |     |     | 0.00 |     |     |     |
| ---- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1136
|     | 0.25 |     |     |     | 0.25 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1137
| 1138 | 0.50 |     |     |     | 0.50 |     |     |     |
| ---- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1139
| 1140 | 0.75 |     |     |     | 0.75 |     |     |     |
| ---- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1141
|     | 1.00 |     |     |     | 1.00 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1142
|     | 8 0 | 1 5 9 | 6 4 7 | 2 3 | 1 2 | 3 4 7 | 8 9 0 | 6 5 |
| --- | --- | ----- | ----- | --- | --- | ----- | ----- | --- |
1143
|     |     | Target Class |     |     |     | Target Class |     |     |
| --- | --- | ------------ | --- | --- | --- | ------------ | --- | --- |
1144
1145
Figure10.BoxplotsoftargetdifferencebytargetclassafterinterventionforunfilteredCNNsMNIST,Fashion-MNIST,CIFAR-10,and
1146 SVHNonthetestset.Orderedbymeantargetdifference;outliersomitted.Upwardsofthereddottedlinemarkssuccessfulintervention.
1147
1148
1149
1150
1151
1152
1153
1154
21

UtilizingWeightSpaceLearningforData-FreeModelEditing
1155
1156
1157
1158
1159
1160
1161
1162
1163
|     |     | MNIST |     |     |     | Fashion-MNIST |     |     |
| --- | --- | ----- | --- | --- | --- | ------------- | --- | --- |
1164
|     | 1.00 |     |     |     | 1.00 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1165
| 1166 | 0.75 |     |     |     | 0.75 |     |     |     |
| ---- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1167
|      | 0.50              |     |     |     | 0.50              |     |     |     |
| ---- | ----------------- | --- | --- | --- | ----------------- | --- | --- | --- |
| 1168 | ecnereffiD tegraT |     |     |     | ecnereffiD tegraT |     |     |     |
1169
|     | 0.25 |     |     |     | 0.25 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1170
1171
|     | 0.00 |     |     |     | 0.00 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1172
| 1173 | 0.25 |     |     |     | 0.25 |     |     |     |
| ---- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1174
|     | 0.50 |     |     |     | 0.50 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1175
1176
|     | 0.75 |     |     |     | 0.75 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1177
1178
|     | 1.00 |     |     |     | 1.00 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1179
|      | 2 9 | 1 8 4        | 6 5 3 | 0 7 | 4 2 | 5 7 9        | 0 3 6 | 1 8 |
| ---- | --- | ------------ | ----- | --- | --- | ------------ | ----- | --- |
| 1180 |     | Target Class |       |     |     | Target Class |       |     |
1181
| 1182 |      | CIFAR-10 |     |     |      | SVHN |     |     |
| ---- | ---- | -------- | --- | --- | ---- | ---- | --- | --- |
|      | 1.00 |          |     |     | 1.00 |      |     |     |
1183
1184
|     | 0.75 |     |     |     | 0.75 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1185
| 1186 | 0.50              |     |     |     | 0.50              |     |     |     |
| ---- | ----------------- | --- | --- | --- | ----------------- | --- | --- | --- |
|      | ecnereffiD tegraT |     |     |     | ecnereffiD tegraT |     |     |     |
1187
|     | 0.25 |     |     |     | 0.25 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1188
1189
| 1190 | 0.00 |     |     |     | 0.00 |     |     |     |
| ---- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1191
|     | 0.25 |     |     |     | 0.25 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1192
| 1193 | 0.50 |     |     |     | 0.50 |     |     |     |
| ---- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1194
| 1195 | 0.75 |     |     |     | 0.75 |     |     |     |
| ---- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1196
|     | 1.00 |     |     |     | 1.00 |     |     |     |
| --- | ---- | --- | --- | --- | ---- | --- | --- | --- |
1197
|     | 0 8 | 1 9 6 | 2 3 5 | 4 7 | 1 2 | 8 3 4 | 7 9 0 | 6 5 |
| --- | --- | ----- | ----- | --- | --- | ----- | ----- | --- |
1198
|     |     | Target Class |     |     |     | Target Class |     |     |
| --- | --- | ------------ | --- | --- | --- | ------------ | --- | --- |
1199
1200
Figure11.BoxplotsoftargetdifferencebytargetclassafterinterventionforonlyfilteredCNNsMNIST,Fashion-MNIST,CIFAR-10,and
1201 SVHNonthetestset.Orderedbymeantargetdifference;outliersomitted.Upwardsofthereddottedlinemarkssuccessfulintervention.
1202
1203
1204
1205
1206
1207
1208
1209
22

UtilizingWeightSpaceLearningforData-FreeModelEditing
1210 F.Extendingtheexperimentation
1211
Themostcommoncriticismvoicedbyreviewerswasabouttheextentoftheexperimentation. Theybelieved:
1212
1213
1214 1. Ourmetanetworkwastrainedonazooofverysmall’toy’CNNstrainedonsimpleclassificationdatasets. Thefieldof
1215 weightspacelearninghasmaturedinthepastfewyearstotheextentthatmanydownstreamtaskshavebeenperformed
1216 proficientlyonlargercomputervisionmodels(LargerCNNsandResNets)2.
1217
2. Comparisonwithbenchmarkswasnotextensiveenough. Weshouldpreferablychooseunlearningmethodologies
1218
whichwouldbeafaircomparisontoourmethodbyworkingwiththesameassumptionsandconstraints.
1219
1220
1221 F.1.Largerbasemodels
1222
AgoodcandidatemetanetworkfortestingourmethodologyonlargerbasemodelscouldbeSANE(Schu¨rholtetal.,2024).
1223
Inordertoachievesuccessthefollowingstepsmustbecompleted:
1224
1225
1. Reasonwhetherunlearningviagradientsw.r.t. theinputcanworkonSANE(specificallygiventhetokenizationofthe
1226
weights,andthatnotallweightsareincorporatedinaforwardpass,butratherasmaller‘window’).
1227
1228
2. EvaluatethemodelsthemodelsintheCIFAR-10CNNmodelzooandsavetheper-classrecallscores. Thesemodels
1229
have3convolutionand2denselayersand 12,000parameterspermodel,already2.4xincreaseinCNNsizecompared
1230
toourpreviousexperiments.
1231
1232 3. PretrainaSANEautoencoderonthisdataset.
1233
4. TrainapredictionheadonthisSANEmodelinstancetopredictper-classrecall
1234
1235
5. PerformunlearningviaAlgorithm1.
1236
1237
Therearemanystepsatwhichthisplancanfail. I’vegotsomealternativesinmyheadbutlet’sgettothosewhenitturnsout
1238
tobeneeded.
1239
1240
1241
1242
1243
1244
1245
1246
1247
1248
1249
1250
1251
1252
1253
1254
1255
1256
1257
1258
1259
1260
1261
1262
2AndrecentlyaViTModelZoowiththepurposeofweightspacelearninghasbeenreleased(Falketal.,2025).However,researchon
1263
weightspacelearningonTransformerarchitecturesisstillveryearlystage.
1264
23