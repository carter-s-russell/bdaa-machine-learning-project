import gradio as gr
import time
import random

def run_densenet_inference(patient_case):
    # Simulate the time it takes for a DenseNet121 model to extract features
    time.sleep(2.5) 
    
    # Generate simulated diagnostic rationales based on the selected boundary case
    if "Glioma" in patient_case:
        predicted_class = "Glioma"
        confidence = round(random.uniform(94.5, 98.2), 2)
        rationale = """
        ### Diagnostic Rationale:
        * **Feature Extraction:** DenseNet identified irregular, diffuse topological boundaries.
        * **Texture Analysis:** High-density raw tissue texture detected in the temporal region.
        * **Bias Check:** Focal Loss confidence threshold exceeded; Meningioma morphology explicitly ruled out.
        * **Status:** Hard rules not triggered. Proceed with clinical review.
        """
        color = "red"
        
    elif "Clear Meningioma" in patient_case:
        predicted_class = "Meningioma"
        confidence = round(random.uniform(98.0, 99.9), 2)
        rationale = """
        ### Diagnostic Rationale:
        * **Feature Extraction:** DenseNet identified clear, well-defined tumor margins.
        * **Location:** Surface-level attachment consistent with standard Meningioma cases.
        * **Status:** High confidence baseline case.
        """
        color = "red"
        
    elif "Pituitary" in patient_case:
        predicted_class = "Pituitary Tumor"
        confidence = round(random.uniform(92.1, 97.5), 2)
        rationale = """
        ### Diagnostic Rationale:
        * **Feature Extraction:** Localized mass detected at the base of the brain.
        * **Spatial Analysis:** Sella turcica region shows abnormal expansion.
        * **Bias Check:** DenseNet feature reuse successfully separated pituitary texture from neighboring tissues.
        * **Status:** Boundary case successfully resolved.
        """
        color = "red"
        
    else:
        predicted_class = "Healthy Control (No Tumor)"
        confidence = round(random.uniform(97.0, 99.5), 2)
        rationale = """
        ### Diagnostic Rationale:
        * **Feature Extraction:** Symmetrical brain morphology detected.
        * **Texture Analysis:** No abnormal density regions or irregular textures identified.
        * **Status:** Clean scan. No further action required.
        """
        color = "green"

    # Format the final output box
    final_output = f"""
    ## Predicted Classification: <span style='color:{color}'>{predicted_class}</span>
    **Model Confidence:** {confidence}%
    
    ---
    {rationale}
    """
    return final_output

# Build the professional UI
with gr.Blocks(theme=gr.themes.Monochrome()) as demo:
    gr.Markdown("# Deep Learning MRI Diagnostic Co-Pilot")
    gr.Markdown("### Powered by DenseNet121 and Focal Loss")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Patient Input")
            case_dropdown = gr.Dropdown(
                choices=[
                    "Case 1: Suspected Glioma (Boundary Case)", 
                    "Case 2: Clear Meningioma", 
                    "Case 3: Suspected Pituitary (Boundary Case)", 
                    "Case 4: Healthy Brain Scan"
                ],
                label="Select MRI Patient Case",
                value="Case 1: Suspected Glioma (Boundary Case)"
            )
            analyze_btn = gr.Button("Run DenseNet Inference", variant="primary")
            
        with gr.Column(scale=2):
            gr.Markdown("### Model Output & Rationale")
            output_display = gr.Markdown("Awaiting scan input...")

    # Connect the button to the Python logic
    analyze_btn.click(
        fn=run_densenet_inference,
        inputs=case_dropdown,
        outputs=output_display
    )

# Launch the local web server
if __name__ == "__main__":
    demo.launch()