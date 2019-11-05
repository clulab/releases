import React from 'react';
import { API_ROOT } from './api-config';
import {PaneLeft, PaneRight} from './components/Pane'
import ModelInput from './ModelInput'
import ModelOutput from './ModelOutput'


class ModelComponent extends React.Component {
    constructor(props) {
      super(props);

      this.state = {
        outputState: "empty",  // valid values: "working", "empty", "received", "error"
        responseData: null,
        claimLabels: [],
        evidenceLabels: []
      };

      this.runModel = this.runModel.bind(this);
    }

    runModel(inputs) {
      this.setState({outputState: "working"});
      let claim_labels = inputs['claim'].split(' ');
      let evidence_labels = inputs['evidence'].split(' ');

      fetch(`${API_ROOT}/predict`, {
        method: 'POST',
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(inputs)
      }).then((response) => {
        return response.json();
      }).then((json) => {
        this.setState({responseData: json, claimLabels: claim_labels, evidenceLabels: evidence_labels, outputState: 'received'})
      }).catch((error) => {
        console.error(error);
        this.setState({outputState: "error"});
      });
    }

    render() {
      const { outputState, responseData, claimLabels, evidenceLabels } = this.state;

      return (
        <div className="pane-container">
          <div className="pane model">
            <PaneLeft>
              <ModelInput runModel={this.runModel} outputState={this.state.outputState}/>
            </PaneLeft>
            <PaneRight outputState={outputState}>
              <ModelOutput outputs={responseData} claimLabels={claimLabels} evidenceLabels={evidenceLabels}/>
            </PaneRight>
          </div>
        </div>
      );

    }
}

export default ModelComponent;
