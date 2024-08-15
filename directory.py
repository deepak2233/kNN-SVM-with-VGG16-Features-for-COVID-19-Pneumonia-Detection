from graphviz import Digraph

def create_pipeline_visualization():
    dot = Digraph(comment='VGG16 Feature Extraction Pipeline')
    
    # Nodes: Defining each step in the process
    dot.node('A', 'Input: Image Data (X_train, X_test)')
    dot.node('B', 'Preprocessing: Rescale, Resize (224x224)')
    dot.node('C', 'VGG16: Pre-trained on ImageNet')
    dot.node('D', 'Feature Extraction (Block 5 Pooling)')
    dot.node('E', 'Extracted Features (X_train_features, X_test_features)')
    
    # Edges: Defining connections between steps
    dot.edges(['AB', 'BC', 'CD', 'DE'])
    
    # Render the pipeline diagram
    return dot

# Create and render the diagram
pipeline_diagram = create_pipeline_visualization()
pipeline_diagram.render('model_pipeline', format='png', cleanup=True)  # Saves as 'model_pipeline.png'
pipeline_diagram.view()
