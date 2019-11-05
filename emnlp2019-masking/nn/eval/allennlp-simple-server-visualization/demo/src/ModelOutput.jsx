import React from 'react';
import HeatMap from './components/heatmap/HeatMap'
import Collapsible from 'react-collapsible'

class ModelOutput extends React.Component {

  formatProb = (num) => num.toFixed(2);

  render() {

    const { outputs, claimLabels, evidenceLabels } = this.props;
	
    // TODO: `outputs` will be the json dictionary returned by your predictor.  You can pull out
    // whatever you want here and visualize it.  We're giving some examples of different return
    // types you might have.  Change names for data types you want, and delete anything you don't
    // need.
    // var string_result_field = outputs['string_result_field'];
    // // This is a 1D attention array, which we need to make into a 2D matrix to use with our heat
    // // map component.
    // var attention_data = outputs['attention_data'].map(x => [x]);
    // // This is a 2D attention matrix.
    // var matrix_attention_data = outputs['matrix_attention_data'];
    // // Labels for our 2D attention matrix, and the rows in our 1D attention array.
    // var column_labels = outputs['column_labels'];
    // var row_labels = outputs['row_labels'];


    var h2p_attention = outputs['h2p_attention'];
    var p2h_attention = outputs['p2h_attention'];
    var labels = outputs['all_labels'];
    var label_probabilites = outputs['label_probs'];
	var tokenized_claim = outputs['tokenized_claim'];
	var tokenized_evidence = outputs['tokenized_evidence'];
	console.log(tokenized_evidence)
	var new_evc_attention = [];
	new_evc_attention[0] = [];
	var evc_attention = outputs['evidence_cum_weights'];
	evc_attention.forEach((v) => {
		new_evc_attention[0].push(v);
	});

	//console.log(evc_attention);
	//console.log(new_evc_attention);
	//console.log(evidenceLabels);
    // This is how much horizontal space you'll get for the row labels.  Not great to have to
    // specify it like this, or with this name, but that's what we have right now.
    var xLabelWidth = "70px";

    return (
      <div className="model__content">

        <div className="form__field">
          <label>Label probabilities</label>
          <table style={{width: "100%", borderCollapse: 'collapse'}}>
            <tr style={{border: '1px solid #dddddd', padding: '8px', textAlign: 'left'}}>
              {labels.map((label) => {
                return <td style={{border: '1px solid #dddddd', padding: '8px', textAlign: 'left'}}>{label}</td>
              })
              }
            </tr>
            <tr style={{border: '1px solid #dddddd', padding: '8px', textAlign: 'left'}}>
              {label_probabilites.map((probs) => {
                return <td style={{border: '1px solid #dddddd', padding: '8px', textAlign: 'left'}}>{this.formatProb(probs)}</td>
              })
              }
            </tr>
          </table>
        </div>

        <div className="form__field">
          <Collapsible trigger="H2P attention map" open={true}>
            <HeatMap xLabels={tokenized_evidence} yLabels={tokenized_claim} data={h2p_attention} xLabelWidth={xLabelWidth} />
          </Collapsible>
		  <Collapsible trigger="Evidence words cumulative attention weights" open={true}>
                <HeatMap xLabels={tokenized_evidence} yLabels={['Rl']} data={new_evc_attention} xLabelWidth={xLabelWidth} />
            </Collapsible>
		  <Collapsible trigger="P2H attention map" open={true}>
            <HeatMap xLabels={tokenized_claim} yLabels={tokenized_evidence} data={p2h_attention} xLabelWidth={xLabelWidth} />
          </Collapsible>
		  
        </div>

      </div>
    );
  }
}

export default ModelOutput;
