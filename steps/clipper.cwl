class: CommandLineTool
cwlVersion: v1.2
id: Ateamclipper
label: clip A-team
doc: |
    Simulates data for the A-team sources based off a skymodel,
    and flags the visibilities of the input MeasuremenSet where
    the model data exceeds the threshold for LBA (50 janskys)
    or HBA (5 janskys).

baseCommand: DP3
arguments:
  - steps=[filter,clipper,counter]
  - clipper.usechannelfreq=False
  - clipper.operation=replace
  - clipper.beamproximitylimit=2000
  - filter.baseline=[CR]S*&
  - filter.remove=False
  - counter.savetojson=True
  - counter.jsonfilename=Ateamclipper.json
  - msout=.
  - msout.storagemanager=dysco
  - msout.storagemanager.databitrate=0

inputs:
  - id: msin
    type: Directory
    inputBinding:
      position: 0
      prefix: msin=
      separate: false
      shellQuote: false
    doc: Input data in MeasurementSet format.

  - id: msin_datacolumn
    type: string?
    default: DATA
    inputBinding:
      position: 0
      prefix: msin.datacolumn=
      separate: false
      shellQuote: false
    doc: |
        Data column of the MeasurementSet
        from which input data is read.

  - id: sources
    type: string[]?
    default:
      - VirA_Gaussian
      - CygA_Gaussian
      - CasA_Gaussian
      - TauA_Gaussian
    inputBinding:
      position: 0
      prefix: clipper.sources=
      separate: false
      itemSeparator: ','
      valueFrom: '"$(self)"'
      shellQuote: false
    doc: |
        Labels of the skymodel patches to
        use to simulate visibilities.

  - id: skymodel
    type: File?
    default:
      location: /usr/local/share/linc/skymodels/A-Team.skymodel
      class: File
    inputBinding:
      position: 0
      separate: false
      prefix: clipper.sourcedb=

  - id: usebeammodel
    type: boolean?
    default: true
    inputBinding:
      position: 0
      prefix: clipper.usebeammodel=True
      shellQuote: false
    doc: |
        Determines whether to use the beam model.

  - id: max_dp3_threads
    type: int?
    default: 5
    inputBinding:
      prefix: numthreads=
      separate: false
      shellQuote: false
    doc: The number of threads per DP3 process.

  - id: number_cores
    type: int?
    default: 12
    doc: |
     Number of cores to use per job for tasks with
     high I/O or memory.

requirements:
  - class: InplaceUpdateRequirement
    inplaceUpdate: true
  - class: InitialWorkDirRequirement
    listing:
      - entry: $(inputs.msin)
        writable: true
  - class: InlineJavascriptRequirement
  - class: ShellCommandRequirement

outputs:
  - id: msout
    doc: Output data in MeasurementSet format.
    type: Directory
    outputBinding:
      glob: $(inputs.msin.basename)

  - id: logfile
    type: File[]
    outputBinding:
      glob: clipper*.log
    doc: |
        The files containing the stdout
        and stderr from the step.

  - id: output
    type: File
    outputBinding:
      glob: Ateamclipper.json
    doc: A text file containing flagging fraction statistics.

hints:
  - class: ResourceRequirement
    coresMin: $(inputs.number_cores)
  - class: DockerRequirement
    dockerPull: vlbi-cwl

stdout: clipper.log
stderr: clipper_err.log
