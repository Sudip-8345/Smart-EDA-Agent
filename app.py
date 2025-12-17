import os, traceback, io, asyncio
import pandas as pd
from save_figures import save_fig
from visualize import generate_visuals, cleanup_files
from DataFrame_to_Str import df_info_string
from ai_text_analysis import ai_text_analysis
from ai_vision_analysis import ai_vision_analysis
import chainlit as cl
import matplotlib
matplotlib.use('Agg')

@cl.on_chat_start
async def start():
    await cl.Message(content="Hello! Upload a CSV file to begin data analysis.").send()
    files = await cl.AskFileMessage(
        content="Please upload your CSV file.", 
        accept=["text/csv", ".csv"],
        max_size_mb=10
    ).send()
    if not files:
        await cl.Message(content="No file uploaded. Please try again.").send()
        return

    processing_msg = await cl.Message(content="Processing your file...").send()
    
    try:
        file = files[0]
        df = pd.read_csv(file.path)
        
        if df.empty:
            await cl.Message(content="The uploaded CSV file is empty. Please upload a valid file.").send()
            return

        cl.user_session.set("dataframe", df)
        info = df_info_string(df)
        await cl.Message(content=f"**DataFrame Info:**\n{info}").send()

        plan = await ai_text_analysis('plan', info)
        await cl.Message(content=f"**Analysis Plan:**\n{plan}").send()

        processing_msg.content = "Generating visualizations..."
        await processing_msg.update()
        
        visuals, saved_files = generate_visuals(df)
        
        # Send each visualization as an image
        for title, img_path in visuals:
            elements = [cl.Image(name=title, path=img_path, display="inline")]
            await cl.Message(content=f"**{title}**", elements=elements).send()
        
        if visuals:
            await cl.Message(content="⏳ Analyzing visualizations with AI...").send()
            
            # Analyze top 3 most important visualizations
            priority_visuals = [v for v in visuals if any(k in v[0] for k in ['Correlation', 'Histogram', 'Boxplot'])][:3]
            if not priority_visuals:
                priority_visuals = visuals[:3]
            
            vision_results = await ai_vision_analysis(priority_visuals, max_images=3)
            for title, analysis in vision_results:
                await cl.Message(content=f"**AI Analysis - {title}:**\n{analysis}").send()
            
            final = await ai_text_analysis('final', info)
            await cl.Message(content=f"**Final Insights:**\n{final}").send()
        
        processing_msg.content = "✅ Analysis complete."
        await processing_msg.update()
        cleanup_files(saved_files)

    except Exception as e:
        error_trace = traceback.format_exc()
        await cl.Message(content=f"An error occurred during processing:\n{e}\n\n{error_trace}").send()