import React from 'react';
import Button from './components/Button'
import ModelIntro from './components/ModelIntro'

const examples = [
  {
    claim: "long text input for example 1",
    evidence: "short text input for example 1"
  },
  {
    claim: "long text input for example 2",
    evidence: "short text input for example 2"
  },
  {
    claim: "long text input for example 3",
    evidence: "short text input for example 3"
  }
];

function summarizeExample(example) {
  return example.claim.substring(0, 60);
}

const title = "Your Model Name";
const description = (
  <span>
  If you want a description of what this demo is showing, you can put that here.  Or just leave this
  description empty if you don't need it.
  </span>
);

class ModelInput extends React.Component {
  constructor(props) {
    super(props);
    this.handleListChange = this.handleListChange.bind(this);
    this.onClick = this.onClick.bind(this);
  }

  handleListChange(e) {
    if (e.target.value !== "") {
      this.claim.value = examples[e.target.value].claim
      this.evidence.value = examples[e.target.value].evidence
    }
  }

  onClick() {
    const { runModel } = this.props;
    runModel({claim: this.claim.value, evidence: this.evidence.value});
  }

  render() {
    const { outputState } = this.props;
    return (
      <div className="model__content">
        <ModelIntro title={title} description={description} />
        <div className="form__instructions"><span>Enter text or</span>
          <select disabled={outputState === "working"} onChange={this.handleListChange}>
              <option value="">Choose an example...</option>
              {examples.map((example, index) => {
                return (
                    <option value={index} key={index}>{summarizeExample(example) + "..."}</option>
                );
              })}
          </select>
        </div>
        <div className="form__field">
          <label>Claim</label>
          <textarea ref={(x) => this.claim = x} type="text" autoFocus="true"></textarea>
        </div>
        <div className="form__field">
          <label>Evidence</label>
          <input ref={(x) => this.evidence = x} type="text"/>
        </div>
        <div className="form__field form__field--btn">
          <Button enabled={outputState !== "working"} onClick={this.onClick} />
        </div>
      </div>
    );
  }
}

export default ModelInput;
