'''Short script to keep well-formatted plots'''

import matplotlib.pyplot as plt
import seaborn as sns

def logit_plot(trials_df, title="", ax=None, show_legend=True):
   
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    
    cp = sns.color_palette("coolwarm", n_colors=4)
    color_map = {
        'A1,R=0': cp[1], 'A1,R=1': cp[0],
        'A2,R=0': cp[2], 'A2,R=1': cp[3]
    }
    
    sns.scatterplot(
        data=trials_df, x='logit_value', y='logit_change',
        hue='trial_type', palette=color_map, ax=ax, legend=False 
    )

    # --- CUSTOM GRID LEGEND TOGGLE ---
    if show_legend:
        # Move x_base > 1.0 to put it to the right of the axes
        # We also need to turn off clipping so the text is visible outside the box
        x_base, y_base = -0.6, 0.8 
        x_space, y_space = 0.12, 0.12

        # Column Headers
        ax.text(x_base, y_base + y_space, '$A_1$', transform=ax.transAxes, 
                ha='center', fontsize=14)
        ax.text(x_base + x_space, y_base + y_space, '$A_2$', transform=ax.transAxes, 
                ha='center', fontsize=14)

        rows = [('R=0', 'A1,R=0', 'A2,R=0'), 
                ('R=1', 'A1,R=1', 'A2,R=1')]

        for i, (label, type1, type2) in enumerate(rows):
            y_pos = y_base - (i * y_space)
            # Row label
            ax.text(x_base - x_space, y_pos, f'${label}$', transform=ax.transAxes, 
                    va='center', ha='right', fontsize=14)
            # Dots
            ax.scatter(x_base, y_pos, color=color_map[type1], 
                       transform=ax.transAxes, s=120, clip_on=False)
            ax.scatter(x_base + x_space, y_pos, color=color_map[type2], 
                       transform=ax.transAxes, s=120, clip_on=False)

    # Styling
    ax.axhline(0, color='grey', linestyle='--')
    ax.set(xlabel='Logit', ylabel='Logit change', title=title)
    max_val = trials_df[['logit_value', 'logit_change']].abs().max().max()

    # 2. Add a little "breathing room" (e.g., 10%)
    buffer = max_val * 1.1

    # 3. Set symmetric limits
    ax.set_xlim(-buffer, buffer)
    ax.set_ylim(-buffer, buffer)

    # 4. Ensure it stays square
    ax.set_aspect('equal', adjustable='box')
    # If the legend is outside, we often need to adjust the layout so it isn't cut off
    if show_legend:
        plt.tight_layout(rect=[0.5, 0, 1.5, 1]) 
        
    return ax