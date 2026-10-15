import PastePanel from './PastePanel.jsx'
import ServedModel from './ServedModel.jsx'
import TrainPanel from './TrainPanel.jsx'

// The Model tab: what the HW2 network is, how it trained (monitor), training
// a new one interactively, and trying it on any text. Each part is a section
// in this frame.
export default function ModelView({ status, pasteText, onPasteText, pasteResult, onPasteResult }) {
  return (
    <div className="panel modelview">
      <div className="mv-head">
        <div className="dtype">Model · HW2</div>
        <h2 className="mv-title">Sentence types: fact / policy / testimony / value</h2>
        <p className="mv-note">
          A frozen MiniLM sentence encoder feeds a small neural network (one hidden layer, 49,796 weights)
          trained on CDCP — citizen comments on a CFPB debt-collection rule. Test error on CDCP is 0.292,
          against 0.327 for a TF-IDF + logistic-regression baseline.
        </p>
        <p className="mv-warn">
          <b>Not validated on these submissions.</b> It predicts policy for 5.2% of RFI sentences against
          11–12% in its training data, and labels most organizational demands as value. Treat its output
          on the corpus as a demonstration, not a finding.
        </p>
        <p className="mv-status">
          {status === null ? (
            'Checking the backend…'
          ) : status.available ? (
            <>
              <span className="live">live</span> · the backend has the model loaded ({status.model.name})
            </>
          ) : (
            <span className="backend-off">Prediction is off: {status.reason}</span>
          )}
        </p>
      </div>

      <section className="mv-section" aria-labelledby="mv-served">
        <h3 id="mv-served" className="mv-section-title">
          Training monitor — the served model
        </h3>
        <ServedModel status={status} />
      </section>

      <section className="mv-section" aria-labelledby="mv-train">
        <h3 id="mv-train" className="mv-section-title">
          Interactive training — train your own
        </h3>
        <TrainPanel status={status} />
      </section>

      <section className="mv-section" aria-labelledby="mv-try">
        <h3 id="mv-try" className="mv-section-title">
          Try the model on any text
        </h3>
        <PastePanel
          status={status}
          text={pasteText}
          onText={onPasteText}
          result={pasteResult}
          onResult={onPasteResult}
        />
      </section>
    </div>
  )
}
