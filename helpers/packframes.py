#!/usr/bin/env python3
"""
PackFrames - A Pac-Man clone with FrameVision branding
Features semi-transparent logo overlays and classic arcade gameplay
"""

import pygame
import random
import os
import math
import json
from typing import List, Tuple, Optional
from datetime import datetime

# Initialize Pygame
pygame.init()

# Game constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
PINK = (255, 192, 203)
CYAN = (0, 255, 255)
ORANGE = (255, 165, 0)
BLUE = (0, 0, 255)
GREY = (128, 128, 128)
DARK_BLUE = (0, 0, 50)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 100, 0)

# Font directory for logos
LOGOS_DIR = "/workspace/presets/startup/"
HIGHSCORES_FILE = "/workspace/helpers/packframes_highscores.json"

# Background constants like FrameRacing
BG_ALPHA = int(255 * 0.35)  # 35% opacity like FrameRacing

def draw_text(surf, text, pos, font=None, color=(240, 240, 240), center=False):
    """Draw text like FrameRacing"""
    if font is None:
        font = pygame.font.Font(None, 22)
    img = font.render(text, True, color)
    rect = img.get_rect()
    if center:
        rect.center = pos
    else:
        rect.topleft = pos
    surf.blit(img, rect)
    return rect

class LogoOverlay:
    """Manages the semi-transparent logo background system using FrameRacing approach"""
    
    def __init__(self):
        self.bg_original = None
        self.bg_surface = None
        self.bg_path = None
        self.bg_last_level_path = None
        self._set_background()
        
    def load_bg_original(self, path):
        """Load background image with alpha (FrameRacing style)"""
        try:
            # Ensure pygame display is initialized before loading images
            if not pygame.display.get_init():
                pygame.display.init()
                
            return pygame.image.load(str(path)).convert_alpha()
        except Exception as e:
            print(f"⚠️ Could not load background image {path}: {e}")
            return None

    def scale_bg_to_window(self, bg_orig):
        """Scale background to window size with alpha (FrameRacing style)"""
        if not bg_orig:
            return None
        scaled = pygame.transform.smoothscale(bg_orig, (SCREEN_WIDTH, SCREEN_HEIGHT))
        scaled.set_alpha(BG_ALPHA)
        return scaled

    def random_bg_original(self, exclude=None):
        """Pick random background excluding current (FrameRacing style)"""
        if not os.path.exists(LOGOS_DIR):
            return None, None
            
        logo_files = [f for f in os.listdir(LOGOS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not logo_files:
            return None, None
            
        # Convert to full paths
        logo_paths = [os.path.join(LOGOS_DIR, f) for f in logo_files]
        
        if exclude and len(logo_paths) > 1:
            logo_paths = [p for p in logo_paths if p != exclude]
            
        choice = random.choice(logo_paths)
        return self.load_bg_original(choice), choice
        
    def _set_background(self, path=None, exclude=None):
        """Set background from path or random (FrameRacing style)"""
        if path is None:
            self.bg_original, self.bg_path = self.random_bg_original(exclude=exclude)
        else:
            self.bg_original = self.load_bg_original(path)
            self.bg_path = path
        self.bg_surface = self.scale_bg_to_window(self.bg_original)
        
    def _rescale_bg(self):
        """Rescale background for window resize (FrameRacing style)"""
        self.bg_surface = self.scale_bg_to_window(self.bg_original)
        
    def init_matrix(self):
        """Initialize simple background - no longer needed, keeping for compatibility"""
        # Matrix background removed for better game visibility
        pass
    
    def select_new_logo(self):
        """Select a new random logo for new levels (FrameRacing style)"""
        self._set_background(path=None, exclude=self.bg_last_level_path)
        self.bg_last_level_path = self.bg_path
    
    def update_matrix(self):
        """Matrix update removed - no longer needed"""
        pass
    
    def draw_matrix_background(self, screen):
        """Matrix background removed for better game visibility"""
        pass
    
    def draw_logo_overlay(self, screen):
        """Draw the semi-transparent logo overlay or fallback to solid background"""
        if self.bg_surface:
            # Draw the logo background
            screen.blit(self.bg_surface, (0, 0))
        else:
            # Fallback: dark blue background when no logo available
            screen.fill(DARK_BLUE)

class HighScoreManager:
    """Manages high score system"""
    
    def __init__(self):
        self.highscores = []
        self.load_highscores()
    
    def load_highscores(self):
        """Load highscores from file"""
        try:
            if os.path.exists(HIGHSCORES_FILE):
                with open(HIGHSCORES_FILE, 'r') as f:
                    self.highscores = json.load(f)
            else:
                self.highscores = []
        except Exception as e:
            print(f"Error loading highscores: {e}")
            self.highscores = []
    
    def save_highscores(self):
        """Save highscores to file"""
        try:
            with open(HIGHSCORES_FILE, 'w') as f:
                json.dump(self.highscores, f, indent=2)
        except Exception as e:
            print(f"Error saving highscores: {e}")
    
    def is_high_score(self, score):
        """Check if score is a high score"""
        if len(self.highscores) < 10:
            return True
        return score > self.highscores[-1]['score']
    
    def add_highscore(self, name, score):
        """Add new high score"""
        entry = {
            'name': name,
            'score': score,
            'date': datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        self.highscores.append(entry)
        self.highscores.sort(key=lambda x: x['score'], reverse=True)
        self.highscores = self.highscores[:10]  # Keep only top 10
        self.save_highscores()
    
    def get_highscores(self):
        """Get top 10 highscores"""
        return self.highscores
    
    def get_player_rank(self, score):
        """Get player's rank if it's a high score"""
        if not self.is_high_score(score):
            return None
        
        temp_scores = sorted([s['score'] for s in self.highscores] + [score], reverse=True)
        return temp_scores.index(score) + 1

class NameInput:
    """Handles player name input for high scores"""
    
    def __init__(self, screen):
        self.screen = screen
        self.input_active = True
        self.player_name = ""
        self.font = pygame.font.Font(None, 48)
        self.small_font = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()
        self.cursor_visible = True
        self.cursor_timer = 0
        
    def handle_event(self, event):
        """Handle input events"""
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN and self.player_name.strip():
                return self.player_name.strip()
            elif event.key == pygame.K_BACKSPACE:
                self.player_name = self.player_name[:-1]
            elif event.key == pygame.K_ESCAPE:
                return "Anonymous"
            elif len(self.player_name) < 20 and event.unicode.isprintable():
                self.player_name += event.unicode
        return None
    
    def update(self):
        """Update cursor visibility"""
        self.cursor_timer += 1
        if self.cursor_timer >= 30:  # Toggle cursor every 30 frames
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = 0
    
    def draw(self):
        """Draw name input interface"""
        # Dark overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(128)
        overlay.fill(BLACK)
        self.screen.blit(overlay, (0, 0))
        
        # Title
        title = self.font.render("NEW HIGH SCORE!", True, YELLOW)
        title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 150))
        self.screen.blit(title, title_rect)
        
        # Instructions
        instructions = self.small_font.render("Enter your name (max 20 chars):", True, WHITE)
        inst_rect = instructions.get_rect(center=(SCREEN_WIDTH // 2, 200))
        self.screen.blit(instructions, inst_rect)
        
        # Name input
        name_text = self.player_name
        if self.cursor_visible:
            name_text += "|"
        name_surface = self.font.render(name_text, True, WHITE)
        name_rect = name_surface.get_rect(center=(SCREEN_WIDTH // 2, 250))
        self.screen.blit(name_surface, name_rect)
        
        # Tips
        tips = [
            "Press ENTER to confirm",
            "Press ESC to use default name",
            "Special characters allowed"
        ]
        
        for i, tip in enumerate(tips):
            tip_surface = self.small_font.render(tip, True, GREY)
            tip_rect = tip_surface.get_rect(center=(SCREEN_WIDTH // 2, 320 + i * 30))
            self.screen.blit(tip_surface, tip_rect)



class Pacman:
    """Pacman character"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 20
        self.speed = 2
        self.direction = "right"
        self.mouth_angle = 0
        
    def move(self, keys):
        """Handle Pacman movement"""
        dx, dy = 0, 0
        
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            dx = -self.speed
            self.direction = "left"
        elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            dx = self.speed
            self.direction = "right"
        elif keys[pygame.K_UP] or keys[pygame.K_w]:
            dy = -self.speed
            self.direction = "up"
        elif keys[pygame.K_DOWN] or keys[pygame.K_s]:
            dy = self.speed
            self.direction = "down"
        
        # Boundary checking
        if 0 <= self.x + dx <= SCREEN_WIDTH - self.size:
            self.x += dx
        if 0 <= self.y + dy <= SCREEN_HEIGHT - self.size:
            self.y += dy
            
        # Animate mouth
        self.mouth_angle = (self.mouth_angle + 0.2) % (2 * math.pi)
        
    def draw(self, screen):
        """Draw Pacman"""
        # Draw mouth animation
        if self.direction == "right":
            mouth_open = 0.3 + 0.2 * math.sin(self.mouth_angle)
            pygame.draw.circle(screen, YELLOW, (int(self.x), int(self.y)), self.size)
            # Draw mouth (black triangle)
            mouth_points = [
                (self.x, self.y),
                (self.x + self.size * 0.8, self.y - self.size * mouth_open),
                (self.x + self.size * 0.8, self.y + self.size * mouth_open)
            ]
            pygame.draw.polygon(screen, BLACK, mouth_points)
        else:
            pygame.draw.circle(screen, YELLOW, (int(self.x), int(self.y)), self.size)

class Ghost:
    """Ghost enemy"""
    
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.size = 18
        self.speed = 1.5
        self.color = color
        self.scatter_target = (x, y)
        
    def move_towards_target(self, target_x, target_y):
        """Move ghost towards target position"""
        dx = target_x - self.x
        dy = target_y - self.y
        
        if abs(dx) > abs(dy):
            if dx > 0:
                self.x += self.speed
            else:
                self.x -= self.speed
        else:
            if dy > 0:
                self.y += self.speed
            else:
                self.y -= self.speed
            
        # Boundary checking
        self.x = max(0, min(SCREEN_WIDTH - self.size, self.x))
        self.y = max(0, min(SCREEN_HEIGHT - self.size, self.y))
    
    def draw(self, screen):
        """Draw ghost"""
        # Simple ghost shape
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.size)
        # Add simple eyes
        pygame.draw.circle(screen, WHITE, (int(self.x - 5), int(self.y - 3)), 3)
        pygame.draw.circle(screen, WHITE, (int(self.x + 5), int(self.y - 3)), 3)
        pygame.draw.circle(screen, BLACK, (int(self.x - 5), int(self.y - 3)), 1)
        pygame.draw.circle(screen, BLACK, (int(self.x + 5), int(self.y - 3)), 1)

class Dot:
    """Food dots"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 3
        self.collected = False
        
    def draw(self, screen):
        """Draw dot"""
        if not self.collected:
            pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), self.size)

class PowerPellet:
    """Power pellets that make ghosts vulnerable"""
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.size = 8
        self.collected = False
        self.pulse_phase = 0
        
    def update(self):
        """Update pulse animation"""
        self.pulse_phase += 0.3
        
    def draw(self, screen):
        """Draw power pellet with pulse effect"""
        if not self.collected:
            pulse_size = int(self.size + 2 * math.sin(self.pulse_phase))
            pygame.draw.circle(screen, WHITE, (int(self.x), int(self.y)), pulse_size)

class Game:
    """Main game class"""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("PackFrames - Retro Arcade Adventure")
        self.clock = pygame.time.Clock()
        
        # Initialize game objects
        self.logo_overlay = LogoOverlay()
        self.pacman = Pacman(100, 100)
        
        # Create ghosts
        self.ghosts = [
            Ghost(400, 300, RED),
            Ghost(350, 250, PINK),
            Ghost(450, 250, CYAN),
            Ghost(400, 350, ORANGE)
        ]
        
        # Create dots and power pellets
        self.dots = []
        self.power_pellets = []
        self.init_dots()
        
        # Game state (FrameRacing style)
        self.state = "START"  # START, RUN, PAUSE, GAMEOVER, MENU, HIGHSCORES, NAME_INPUT
        self.score = 0
        self.level = 1
        self.lives = 3  # Add 3-lives system
        self.ghost_mode = "chase"  # "chase" or "scatter"
        self.running_time = 0.0
        
        # High score system
        self.highscore_manager = HighScoreManager()
        self.name_input = None
        
        # UI buttons like FrameRacing
        self.ui_buttons = {}
        self.init_ui_buttons()
    
    def init_ui_buttons(self):
        """Initialize UI buttons like FrameRacing"""
        # This will be called in draw() method based on current state
        pass
        
    def _draw_right_panel(self, surf):
        """Draw right panel with buttons like FrameRacing"""
        panel_width = 200
        panel = pygame.Rect(SCREEN_WIDTH - panel_width, 0, panel_width, SCREEN_HEIGHT)
        s = pygame.Surface(panel.size, pygame.SRCALPHA)
        s.fill((0, 0, 0, 110))
        surf.blit(s, panel.topleft)
        
        # Buttons
        self.ui_buttons = {}
        btn_y = 10
        btn_w, btn_h = 68, 28
        gap = 12
        x = panel.left + 12
        
        # Determine buttons based on state
        if self.state == "START":
            labels = [("Start", "start"), ("Exit", "exit")]
        elif self.state == "RUN":
            labels = [("Pause", "pause"), ("Exit", "exit")]
        elif self.state == "PAUSE":
            labels = [("Resume", "start"), ("Exit", "exit")]
        elif self.state == "GAMEOVER":
            labels = [("Restart", "start"), ("Exit", "exit")]
        else:
            labels = [("Menu", "menu"), ("Exit", "exit")]
            
        for label, key in labels:
            rect = pygame.Rect(x, btn_y, btn_w, btn_h)
            pygame.draw.rect(surf, (230, 230, 230), rect, border_radius=8)
            text_surface = pygame.font.Font(None, 20).render(label, True, (20, 20, 20))
            text_rect = text_surface.get_rect(center=rect.center)
            surf.blit(text_surface, text_rect)
            self.ui_buttons[key] = rect
            x += btn_w + gap
        
        # High scores section
        font = pygame.font.Font(None, 24)
        draw_text(surf, "Top 10", (panel.centerx, 60), font=font, color=YELLOW, center=True)
        scores = self.highscore_manager.get_highscores()
        y = 80
        for i, score_entry in enumerate(scores[:10]):
            name = score_entry['name'][:12]
            score = score_entry['score']
            txt = f"{i+1:>2}. {name:<12} {score:>5}"
            draw_text(surf, txt, (panel.left + 12, y), font=font)
            y += 18
    
    def init_dots(self):
        """Initialize dots and power pellets on the screen"""
        # Create grid of dots
        for x in range(50, SCREEN_WIDTH - 50, 40):
            for y in range(50, SCREEN_HEIGHT - 50, 40):
                if random.random() < 0.7:  # 70% chance for a dot
                    self.dots.append(Dot(x, y))
        
        # Create power pellets at corners and center
        power_positions = [
            (50, 50), (SCREEN_WIDTH - 50, 50),
            (50, SCREEN_HEIGHT - 50), (SCREEN_WIDTH - 50, SCREEN_HEIGHT - 50),
            (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        ]
        
        for x, y in power_positions:
            self.power_pellets.append(PowerPellet(x, y))
    
    def start_game(self):
        """Start the game (FrameRacing style)"""
        self.state = "RUN"
        self.score = 0
        self.level = 1
        self.lives = 3  # Initialize 3 lives
        self.ghost_mode = "chase"
        self.running_time = 0.0
        self.pacman = Pacman(100, 100)
        self.ghosts = [
            Ghost(400, 300, RED),
            Ghost(350, 250, PINK),
            Ghost(450, 250, CYAN),
            Ghost(400, 350, ORANGE)
        ]
        self.dots = []
        self.power_pellets = []
        self.init_dots()
        self.logo_overlay._set_background()  # Start with random logo
        print("🎮 Game started!")
    
    def toggle_pause(self):
        """Toggle pause state (FrameRacing style)"""
        if self.state == "RUN":
            self.state = "PAUSE"
            print("⏸️ Game paused!")
        elif self.state == "PAUSE":
            self.state = "RUN"
            print("▶️ Game resumed!")
    
    def show_menu(self):
        """Show main menu"""
        self.state = "START"
        print("🏠 Returning to menu")
    
    def show_highscores(self):
        """Show high scores"""
        self.state = "HIGHSCORES"
        print("🏆 High scores displayed")
    
    def quit_game(self):
        """Quit the game"""
        self.state = "QUIT"
        print("👋 Goodbye!")
    
    def check_collisions(self):
        """Check collisions between Pacman and game objects"""
        pacman_rect = pygame.Rect(self.pacman.x - self.pacman.size, 
                                 self.pacman.y - self.pacman.size,
                                 self.pacman.size * 2, self.pacman.size * 2)
        
        # Check dot collection
        for dot in self.dots:
            if not dot.collected:
                dot_rect = pygame.Rect(dot.x - dot.size, dot.y - dot.size,
                                     dot.size * 2, dot.size * 2)
                if pacman_rect.colliderect(dot_rect):
                    dot.collected = True
                    self.score += 10
        
        # Check power pellet collection
        for pellet in self.power_pellets:
            if not pellet.collected:
                pellet_rect = pygame.Rect(pellet.x - pellet.size, pellet.y - pellet.size,
                                         pellet.size * 2, pellet.size * 2)
                if pacman_rect.colliderect(pellet_rect):
                    pellet.collected = True
                    self.score += 50
                    self.ghost_mode = "scatter"
    
    def check_ghost_collisions(self):
        """Check collisions between Pacman and ghosts"""
        pacman_rect = pygame.Rect(self.pacman.x - self.pacman.size, 
                                 self.pacman.y - self.pacman.size,
                                 self.pacman.size * 2, self.pacman.size * 2)
        
        for ghost in self.ghosts:
            ghost_rect = pygame.Rect(ghost.x - ghost.size, ghost.y - ghost.size,
                                   ghost.size * 2, ghost.size * 2)
            if pacman_rect.colliderect(ghost_rect):
                if self.ghost_mode == "scatter":
                    # Pacman eats ghost
                    self.score += 200
                    ghost.x = SCREEN_WIDTH // 2
                    ghost.y = SCREEN_HEIGHT // 2
                else:
                    # Ghost catches Pacman - handle lives system
                    self.lose_life()
                    return
    
    def lose_life(self):
        """Handle losing a life"""
        self.lives -= 1
        print(f"💔 Life lost! Remaining lives: {self.lives}")
        
        if self.lives <= 0:
            # No more lives - game over
            self.end_game()
        else:
            # Reset positions and continue
            self.reset_positions()
    
    def reset_positions(self):
        """Reset Pacman and ghost positions after losing a life"""
        self.pacman.x = 100
        self.pacman.y = 100
        
        # Reset ghosts to center
        for ghost in self.ghosts:
            ghost.x = SCREEN_WIDTH // 2
            ghost.y = SCREEN_HEIGHT // 2
        
        self.ghost_mode = "chase"  # Reset mode
        print(f"🔄 Positions reset - Lives: {self.lives}")
    
    def end_game(self):
        """End the game and check for high score (FrameRacing style)"""
        print(f"💀 Game Over! Final Score: {self.score}")
        
        if self.highscore_manager.is_high_score(self.score):
            rank = self.highscore_manager.get_player_rank(self.score)
            print(f"🎉 New High Score! Rank: #{rank}")
            self.state = "NAME_INPUT"
            self.name_input = NameInput(self.screen)
        else:
            self.state = "GAMEOVER"
    
    def restart_game(self):
        """Restart the game"""
        self.start_game()
    
    def next_level(self):
        """Advance to next level"""
        self.level += 1
        self.logo_overlay.select_new_logo()  # Change logo for new level
        self.ghost_mode = "chase"
        
        # Reset Pacman position
        self.pacman.x = 100
        self.pacman.y = 100
        
        # Reset dots and pellets
        self.dots = []
        self.power_pellets = []
        self.init_dots()
        
        print(f"🏁 Level {self.level}! New logo selected.")
    
    def update(self, dt):
        """Update game logic (FrameRacing style)"""
        if self.state != "RUN":
            if self.state == "NAME_INPUT":
                if self.name_input:
                    self.name_input.update()
            return
            
        # Update running time
        self.running_time += dt
            
        # Update power pellet animations
        for pellet in self.power_pellets:
            pellet.update()
        
        # Update ghost positions
        for ghost in self.ghosts:
            if self.ghost_mode == "chase":
                # Ghosts chase Pacman
                ghost.move_towards_target(self.pacman.x, self.pacman.y)
            else:
                # Ghosts scatter
                ghost.move_towards_target(ghost.scatter_target[0], ghost.scatter_target[1])
        
        # Matrix background removed - no longer needed
        
        # Check collisions
        self.check_collisions()
        self.check_ghost_collisions()
        
        # Check if level complete
        remaining_dots = sum(1 for dot in self.dots if not dot.collected)
        remaining_pellets = sum(1 for pellet in self.power_pellets if not pellet.collected)
        
        if remaining_dots == 0 and remaining_pellets == 0:
            self.next_level()
    
    def handle_events(self):
        """Handle pygame events (FrameRacing style)"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.state = "QUIT"
                return
            
            # Handle mouse button clicks for UI buttons
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if "start" in self.ui_buttons and self.ui_buttons["start"].collidepoint(pos):
                    if self.state in ["START", "GAMEOVER"]:
                        self.start_game()
                    elif self.state == "PAUSE":
                        self.state = "RUN"
                    return
                if "pause" in self.ui_buttons and self.ui_buttons["pause"].collidepoint(pos):
                    if self.state == "RUN":
                        self.state = "PAUSE"
                    return
                if "exit" in self.ui_buttons and self.ui_buttons["exit"].collidepoint(pos):
                    self.state = "QUIT"
                    return
                if "menu" in self.ui_buttons and self.ui_buttons["menu"].collidepoint(pos):
                    self.state = "START"
                    return
            
            # Handle keyboard events
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.state == "RUN":
                        self.state = "PAUSE"
                    elif self.state == "PAUSE":
                        self.state = "RUN"
                    elif self.state in ["START", "GAMEOVER"]:
                        self.state = "QUIT"
                    return
                
                if self.state == "START":
                    if event.key == pygame.K_RETURN:
                        self.state = "RUN"
                
                elif self.state == "RUN":
                    if event.key == pygame.K_SPACE:
                        self.toggle_pause()
                
                elif self.state == "PAUSE":
                    if event.key == pygame.K_SPACE:
                        self.state = "RUN"
                
                elif self.state == "NAME_INPUT":
                    if event.key == pygame.K_RETURN:
                        if self.name_input:
                            result = self.name_input.handle_event(event)
                            if result:
                                self.highscore_manager.add_highscore(result, self.score)
                                print(f"🏆 High score saved: {result} - {self.score}")
                                self.state = "GAMEOVER"
                                self.name_input = None
                
                elif self.state == "GAMEOVER":
                    if event.key == pygame.K_r:
                        self.start_game()
                    elif event.key == pygame.K_RETURN and self.highscore_manager.is_high_score(self.score):
                        self.state = "NAME_INPUT"
                        self.name_input = NameInput(self.screen)
    

    
    def draw(self):
        """Draw the game using FrameRacing double-buffering approach"""
        # Create world surface like FrameRacing
        world = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        world.fill(BLACK)
        
        # Draw background elements on world
        self.logo_overlay.draw_matrix_background(world)
        if self.logo_overlay.bg_surface:
            world.blit(self.logo_overlay.bg_surface, (0, 0))
        
        # Draw game elements on world
        if self.state in ["RUN", "PAUSE"]:
            # Draw dots
            for dot in self.dots:
                dot.draw(world)
                
            # Draw power pellets
            for pellet in self.power_pellets:
                pellet.draw(world)
            
            # Draw Pacman
            self.pacman.draw(world)
            
            # Draw ghosts
            for ghost in self.ghosts:
                ghost.draw(world)
        
        # Blit world to screen
        self.screen.fill(BLACK)
        self.screen.blit(world, (0, 0))
        
        # Draw right panel with buttons and high scores
        self._draw_right_panel(self.screen)
        
        # Draw UI elements and game state messages
        self.draw_game_ui()
        
        # Update display
        pygame.display.flip()
        
    def draw_game_ui(self):
        """Draw game UI elements based on state"""
        font = pygame.font.Font(None, 36)
        
        # Draw score and lives at top
        score_text = font.render(f"Score: {self.score}", True, YELLOW)
        self.screen.blit(score_text, (10, 10))
        
        lives_text = font.render(f"Lives: {self.lives}", True, YELLOW)
        self.screen.blit(lives_text, (10, 50))
        
        level_text = font.render(f"Level: {self.level}", True, YELLOW)
        self.screen.blit(level_text, (10, 90))
        
        if self.state == "START":
            # Title screen
            title_font = pygame.font.Font(None, 72)
            title = title_font.render("PACKFRAMES", True, YELLOW)
            title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 200))
            self.screen.blit(title, title_rect)
            
            subtitle_font = pygame.font.Font(None, 24)
            subtitle = subtitle_font.render("Retro Arcade Adventure", True, WHITE)
            subtitle_rect = subtitle.get_rect(center=(SCREEN_WIDTH // 2, 240))
            self.screen.blit(subtitle, subtitle_rect)
            
            # Instructions
            instructions = [
                "Avoid ghosts, collect all dots!",
                "Power pellets make ghosts vulnerable",
                "WASD or Arrow Keys to move",
                "SPACE: Pause during game",
                "ESC: Pause during game, Exit from menu",
                "Press ENTER to start"
            ]
            for i, instruction in enumerate(instructions):
                text = pygame.font.Font(None, 24).render(instruction, True, WHITE)
                text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, 300 + i * 25))
                self.screen.blit(text, text_rect)
                
        elif self.state in ["RUN", "PAUSE"]:
            # Game HUD
            score_text = font.render(f"Score: {self.score}", True, WHITE)
            level_text = font.render(f"Level: {self.level}", True, WHITE)
            mode_text = font.render(f"Mode: {self.ghost_mode.title()}", True, WHITE)
            
            self.screen.blit(score_text, (10, 10))
            self.screen.blit(level_text, (10, 50))
            self.screen.blit(mode_text, (10, 90))
            
            # Pause message
            if self.state == "PAUSE":
                pause_font = pygame.font.Font(None, 72)
                paused_text = pause_font.render("PAUSED", True, WHITE)
                text_rect = paused_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
                self.screen.blit(paused_text, text_rect)
                
                resume_font = pygame.font.Font(None, 24)
                resume_text = resume_font.render("Press SPACE or click Resume to continue", True, WHITE)
                resume_rect = resume_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
                self.screen.blit(resume_text, resume_rect)
                
        elif self.state == "GAMEOVER":
            # Game over screen
            game_over_font = pygame.font.Font(None, 72)
            game_over_text = game_over_font.render("GAME OVER", True, WHITE)
            text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, 200))
            self.screen.blit(game_over_text, text_rect)
            
            final_score_font = pygame.font.Font(None, 36)
            final_score_text = final_score_font.render(f"Final Score: {self.score}", True, YELLOW)
            score_rect = final_score_text.get_rect(center=(SCREEN_WIDTH // 2, 250))
            self.screen.blit(final_score_text, score_rect)
            
            # Check if it's a high score
            if self.highscore_manager.is_high_score(self.score):
                rank = self.highscore_manager.get_player_rank(self.score)
                rank_text = final_score_font.render(f"NEW HIGH SCORE! Rank: #{rank}", True, GREEN)
                rank_rect = rank_text.get_rect(center=(SCREEN_WIDTH // 2, 290))
                self.screen.blit(rank_text, rank_rect)
                
        elif self.state == "NAME_INPUT":
            # Name input overlay
            if self.name_input:
                self.name_input.draw()
                
        elif self.state == "HIGHSCORES":
            # High scores screen (background handled by world surface)
            title_font = pygame.font.Font(None, 48)
            title = title_font.render("HIGH SCORES", True, YELLOW)
            title_rect = title.get_rect(center=(SCREEN_WIDTH // 2, 100))
            self.screen.blit(title, title_rect)
            
            # Back instruction
            back_text = pygame.font.Font(None, 24).render("Click Menu button to return", True, WHITE)
            back_rect = back_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 50))
            self.screen.blit(back_text, back_rect)
    
    @property
    def game_running(self):
        """Check if game should continue running"""
        return self.state != "QUIT"
    
    def run(self):
        """Main game loop (FrameRacing style)"""
        print("🎮 Welcome to PackFrames!")
        print("🔄 Loading random logo from presets/startup/...")
        
        dt = 0.0
        while self.game_running:
            self.handle_events()
            
            # Handle movement input
            if self.state == "RUN":
                keys = pygame.key.get_pressed()
                self.pacman.move(keys)
            
            # Update game logic
            self.update(dt)
            
            # Draw everything
            self.draw()
            
            # Control frame rate
            dt = self.clock.tick(FPS) / 1000.0
        
        pygame.quit()
        print("👋 Thanks for playing PackFrames!")

def main():
    """Main entry point"""
    print("🎯 Starting PackFrames - FrameVision Pac-Man Clone")
    print("📁 Looking for logo files in:", LOGOS_DIR)
    
    try:
        game = Game()
        game.run()
    except Exception as e:
        print(f"❌ Error starting game: {e}")
        print("💡 Make sure Pygame is installed and logo files exist in presets/startup/")

if __name__ == "__main__":
    main()