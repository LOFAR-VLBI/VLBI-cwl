class: CommandLineTool
cwlVersion: v1.2
id: findRefAnt_join
label: Find Reference Antenna join
doc: |
    Selects station data from a dictionary, determines
    and outputs the fraction of flagged data and the
    station name with the least amount of flagged data.

baseCommand: findRefAnt_join.py

arguments:
 - valueFrom: input.json
   prefix: --flagged_fraction_dict

inputs:
    - id: flagged_fraction_dict
      type: string[]?
      default: []
      doc: A list of flagged antennas per MeasurementSet.

    - id: filter_station
      type: string?
      default: '*&'
      inputBinding:
        position: 1
        prefix: --station_filter
      doc: A regular expression pattern for station names to select.

    - id: state
      type: string?
      default: 'NONE'
      inputBinding:
        position: 2
        prefix: --state
      doc: State information for the collection of antenna statistics.

outputs:
  - id: refant
    type: string
    outputBinding:
        loadContents: true
        glob: out.json
        outputEval: $(JSON.parse(self[0].contents).refant)
    doc: |
        The reference antenna, containing
        the least amount of flagged data.

  - id: flagged_fraction_antenna
    type: File
    outputBinding:
      glob: flagged_fraction_antenna.json
    doc: |
        The fraction of flagged data per
        antenna in a CSV format.

  - id: logfile
    type: File
    outputBinding:
      glob: findRefAnt.log
    doc: |
        The files containing the stdout
        and stderr from the step.

requirements:
  - class: InlineJavascriptRequirement
  - class: InitialWorkDirRequirement
    listing:
      - entryname: input.json
        entry: $(inputs.flagged_fraction_dict)

hints:
  - class: DockerRequirement
    dockerPull: vlbi-cwl

stdout: findRefAnt.log
stderr: findRefAnt_err.log
