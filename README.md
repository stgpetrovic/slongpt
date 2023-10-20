# slongpt
Playing with a course to learn transformer basics.

Following the [tutorial on yt](https://www.youtube.com/watch?v=kCc8FmEb1nY), I managed to produce some bard lyrics.
The model trains on [tiny shakespeare](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt)
and learns to produce Shakespeare-like text. Here's 10k tokens (characters).

It's not Hamlet, but warms my hearth.

```bash
// NVIDIA GTX 1950 Mobile
// Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz
step 0: train loss 4.2948, val loss 4.3013
step 500: train loss 1.7599, val loss 1.9029
step 1000: train loss 1.5333, val loss 1.7262
step 1500: train loss 1.4367, val loss 1.6412
step 2000: train loss 1.3731, val loss 1.5947
step 2500: train loss 1.3352, val loss 1.5667
step 3000: train loss 1.3040, val loss 1.5440
step 3500: train loss 1.2697, val loss 1.5356
step 4000: train loss 1.2543, val loss 1.5292
step 4500: train loss 1.2290, val loss 1.5173                                                         
```

DUCHESS OF YORK:                                                                                      
What forsh was I desert in, persadPy him.                                                             
And harping of one it leave lived him as Cominius.                                                    
                                                                                                                                                                                                            
Both:                                                                                                                                                                                                       
We'll much awhician                                                                                   
One to speak you to your evior that young Pabethinks                                                                                                                                                        
should requites his friend: and judgmen, my                                                           
householding held his hand. Didst innotate him.                                                       

BRUTUS:                                                                                               
Nay, good noble Conquent they are of good                                                             
day-lain'd a plags an hour strikening mouth.                                                          
The recompesses with chaithes of man.                                                                 

AUTOLYCUS:                                                                                            
That calls fly upon my raund as harned                                                                
Willingly play I had touch'd, we much fail.                                                           

LUCENTIO:                                                                                             
Beseech yor; this wide but yieldang good                                                              
Complain, he hollain mine ears, Descention,                                                           
Shough, live makes me to be beat, I                                                                   
proper father affront, and does me                                                                    
infanting, grantingmending me are plumpier'd.                                                         

CORIOLANUS:                                                                                           
Say you to of our child news;                                                                         
Moccanspured more up my man; cure the face,                                                           
O thy triumphing mest                                                                                 
By that we there best by camilly: in--Who, impant's beh,                                                                                                                                                    
As nevis and friends wean deeds, are nor raimed                                                       
confessed in libert: pluck upon us agadnoget                                                          
And os nielded yet cannot to be                                                                       
some cannot gentlemen buy to seen a remolaticn.

BENVOLIO:                                                                                             
Thou that you told me any be not of fight.                                                            

DUCHESS OVELUS:                                                                                       
Go to! If the king nor how that lapp is sent.                                                         

ISABELLO:                                                                                             
Wheresay shedegnessip'd that is that?                                                                 
You are as it fellows tomode:                                                                         
Toke it me: 'letter he be set which I desire                                                          
winhipp'd upon with affles that I ne'er unamost                                                       
That I cry than Rome, away.                                                                           
Sirrah, I sue the all it give be a mine.                                                              

BUCKINGHAM:                                                                                           
Jupitor Gaunt is Montague what with our true:                                                         
Hie may be stranged the Misage hot:                                                                   
Unlike pertake in great read oncing Richard's stae.                                                   

NORTHUMBERLAND:                                                                                       
Not that Rome be wards so drinken; so; I will to say 't:                                                                                                                                                    
What now yes your leadly perfirm'd with him                                                           
And by tinners to excuse                                                                              
Him well; as Mantua, he speakest and                                                                  
crepture foright; helms it piece'd the letter                                                         
That looks will do imprept my undererse to the                                                        
matter o' them; companyer.                                                                            

HENRY BOLINGBROKE:                                                                                    
Acquity but all to be anone; look thou frail                                                          
As if a thousand style as was down antellion.                                                         
Cove forth on my land hastpenged                                                                      
From him some deedful lassh shout his dead,                                                           
At this rights helf in the favoth: make fortunent,                                                    
He unsaw nave very faced her make her see:                                                            
Petelling we I in the king the mall we dost                                                           
Whose is noise eye mine exile in the city                                                             
And bloody order enmities have made with livining scure                                                                                                                                                     
Lest to betrain neved, nor night.                                                                     
What should will be dust dead.                                                                        
Lords, were love you twenty yon; garle, then was planet                                                                                                                                                     
Of course could heaven! see! what, hast thou art                                                      
then? Hark loveth, as yours! where is general cleyal to                                                                                                                                                     
Off this the mocks o'er infection of them                                                             
to save it great and, what you are: get me bless.

YORK:                                                                                                 
Tybles 'en I cannot thee and of the son                                                               
Conspected dies the beat of the earth of schook                                                       
burn pening, acquain my mornary prisoner;                                                             
On thinken you both him and join the solf                                                             
To thy sourner London's business, an acquitter foe,                                                   
Hath firmed a man, which, which we were not hear,                                                     
On think out, and scere, lords so, crown'dst block my brother.                                                                                                                                              
Tutuff much will depart your pale Bolingbroke: when it yet                                                                                                                                                  
Pompey were gentlemen! my lords, let him in turns:                                                    
All his honestance that I crowns, for well gent took:                                                                                                                                                       
I'll swear with a woman out more wordds part so:                                                      
Where maltingle say, in my dearing, a poor                                                            
To the faith, conto dispact your power.                                                               
Hown now, comfort! all gives you, that make it fit. I                                                                                                                                                       
said his father?                                                                                      
What he are you, as all his makes delive, he carrents                                                                                                                                                       
Helon so; but live                                                                                    
And that she was I may be smyderaked,                                                                 
As I the Clifford to the fale one blood.                                                              
Take my cup, and so nothing wis that silent does                                                      
I seem thyself.                                                                                       

WESWARWICK:                                                                                           
On, and what he drids makes of his hire,                                                              
Sin soins curser mine reputaring were doom.                                                           
Speak was don you to the Prince.                                                                      
Besides, make it you strength morch'd not on't!                                                       
Who knows so resomething me! My lace,                                                                 
When my cousinans I should even I lived hours:                                                        
It need shrucking mask'd in his mettles                                                               
Confine anointed boister with my notedness,                                                           
And her cleisure to confer i' the throne: 'tis most                                                   
And made on the blow of why coloud dijes                                                              
is handred.

CAMILLO:                                                                                              
Speak it on                                                                                           
The resolution on! The Duke of by angers, boys                                                        
Well threadfaction that's ever at thy vower,                                                          
helmeor in defence with thee urges these drunk?                                                       
The Volsces is his sear'd soverent, and honest better.                                                                                                                                                      
Thou hast mother; and that have strow'd all.                                                          
Forbid my lord,                                                                                       
But will I prepture thee ladk, my cheby?                                                              
Go to defend my lord, uncle; and here's a prayer.                                                     
If thoughts we may sitate, you may play not may do;                                                   
That have common fought that been lived so.                                                           
Give me the conspirance o' the balmt, and rivious                                                     
Donne trull on 't.                                                                                    
We had not of in my sense, at once more of                                                            
CATESBY:                                                                                              
Did not up the clocketed of my high's soul,                                                           
The command mortain with salk and him.                                                                

HENRY BOLINGBROKE:                                                                                    
Will dead, what Dovoli                                                                                
With his senate Lewis down, and year anotherse.                                                       

KING RICHARD III:                                                                                     
Not seek that sword him to could the second                                                           
That myself an addestily, those ladies fade                                                           
Thy from thy wretchen'd.                                                                              

DUKE OF AUMERLE:                                                                                      
In God our queen one, like a second unto your curse paint.                                                                                                                                                  

KING RICHARD II:                                                                                      
Uncle Divorce speak with him, make you gone,                                                          
While he weepp you this restance intended.                                                            

WARWICK:                                                                                              
This name is lieging starve men from service                                                          
That once makes you an, and quied this acts offence                                                   
At brothen's bond may not know                                                                        
For the Dentgomio of Nance                                                                            
Norfolk: great a Pompeyong man: hoa can is                                                            
desire, life, and theu fall'nt make lives: therefore                                                                                                                                                        
thou madst softlet not kiss your friends' business!                                                   
Thou tushs not that trench is in move an, that                                                        
of it? As gone thy crave six, in Montagues there did give my life;                                                                                                                                          
Which at loving you givernme ends                                                                     
by the more standter. Become, come; that you great is a Monturefuse                                                                                                                                         
The rude is nothing claman.

LUCENTIO:                                                                                             
A many it flesh of good night!                                                                        
I pray you, maidly here I proudly in this;                                                            
Had I lend your mouthfully crease you better.                                                         

LUVENCENTIO:                                                                                          
None's so ancess!                                                                                     

LUCENTIO:                                                                                             
When when tven.                                                                                       
A man by my dangerous lord, ere you so!                                                               
We may please their otent                                                                             
An outs had you York bend beggar.                                                                     

FRIAR LAURENCE:                                                                                       
Canspired and desires the hour;                                                                       
And once you together miselance, look you, 'tis not not:                                                                                                                                                    
The hum till better to-night; and speak no gross:                                                     
It raised nor thill be mones!                                                                         
'Tis by rough'd in the reak of smale                                                                  
Etender, such many to the cease be                                                                    
To make one one fiends.                                                                               

YORK:                                                                                                 
Good Capel, thy lips is nob motice;                                                                   
And one is shown it to that hance for sometidents                                                     
Lanst us an issulent scripply whose walls                                                             
First for our restected, feast' men. This day did                                                     
The sent all the wark's yea? I will not buy                                                           
The condust to Irelant a begggary: and                                                                
We should find again. He maystore death I                                                             
When desire we, indeedom weep here for.                                                               

ROMEO:                                                                                                
O, is only child;                                                                                     
And, nor children anot of thartfue to be                                                              
Bight an outrance of his nobone, love's snop;                                                         
Good vass all belic you too his reems too,                                                            
And sinclume him elept have, crall'd foreheved in partice.'                                                                                                                                                 
ISABELLA:                                                                                             
Any in the father.                                                                                    

LADY GELO:                                                                                            
Well, mear you move of, who with have been brought.

DUKE OF AUMERLE:                                                                                      
Stiffs soon and ware but not of alms,                                                                 
And such and aftern blows. His sanctions we nobly restorned,                                                                                                                                                
The vents man truth note of me; fex be respect                                                        
Is once penius, slept the convious to love                                                            
May make the rose in war                                                                              
That humorone a solence, cause that fetch age alive,                                                                                                                                                        
And nor nor humany man; with That we doubtful                                                         
Of ceeding Fleshrest an air, how he can presents as my                                                                                                                                                      
heart as we goess his nriches friends of France: peaces,                                                                                                                                                    
Biankle my join; is they are cancell'd my part:                                                       
My brother were they standards, his grain.                                                            

BRUTUS:                                                                                               
Didst she betice the fence                                                                            
And in suplume corapt to sten me:                                                                     
O, Anonly citle English shall the suin.                                                               

OXFORDOLEO:                                                                                           
I did, now 'tis leave to relp them have in good.                                                      

QURENCE:                                                                                              
And let pine hides laint the sacre                                                                    
Which yet the sten join I stand, I come too much                                                      
Affrance from theMAL:                                                                                 
Understable say you me prove age                                                                      
What remedyly were we determined by the shame,                                                        
Eich injulties as Bolot, tongue Richam, Cates!                                                        
Here one the king What lessing effects of company                                                     
Which that he had provenjoisty and enforces alms                                                      
As friends like all their loov: death he drespect.                                                    

GLOUCESTER:                                                                                           
Bend say, tell the wench.                                                                             
This is the other's son, think when should bind                                                       
You make her hearthe you, let it the subt,                                                            
The migood fled our feather be a shepherd,                                                            
And say 'I spoke come for Peter                                                                       
The strength and secret for reason!' this beneface                                                    
'Tis not heart since to tille the thing arms.                                                         

Third Self give you, unthinking?                                                                      

WARWICK:                                                                                              
Dear sender besred, our king Jorn as well.                                                            

RIVERS:                                                                                               
See you privy first be much to make abite. How                                                        
once cannot attend not? would you hurr be panished,                                                   
temperable in friend him speeds of his?

LUCENTIO:                                                                                             
Sir? how is he hath shamed?                                                                           

Provost:                                                                                              
A bricky hathful                                                                                      
Be spare. That calls you death downfy no same home                                                    
What I say, lord who should wish'd Tybalt                                                             
With this reprehensoe's sound.                                                                        

CAPULET:                                                                                              
Great I'll prove the people respied orders, 'tis doubt                                                                                                                                                      
And shake go to stuff, weep; and thunk in some strength                                                                                                                                                     
windrights no more of the defiances, your power:                                                      
Have you so done as he was pluck and what set                                                         
'Tigs is not I should in my sad.'                                                                     
And what co must till I not live strangel'd at my peaces.                                                                                                                                                   

DUKE VINCENTIO:                                                                                       
Have you still let gue your                                                                           
Loy leave to poise on the joints.                                                                     

LUCENTIO:                                                                                             
How shall meet hence? Is tongue awhile. Found what                                                    
the bleave reuteous numbers presended on? stray you?                                                                                                                                                        
Shere she beil your passion. Then is a dead,                                                          
that's sent of him for your lord,                                                                     
And see the praise truth of his breast him.                                                           

LEONTES:                                                                                              
Went thou give me                                                                                     
To meeter, more condemnant,                                                                           
More man! What was I emplacient lized and crease                                                      
At natiant ne'er cannot.                                                                              

FLORIZEL:                                                                                             
And shrence I near uncle to see thee they wask;                                                       
Dear me reply not they say.                                                                           

RIVERS:                                                                                               
Most cannot deeply an eaglining sength                                                                
That catcher conceit, or curess none present and the                                                                                                                                                        
Divicence thoughts, and bitken? he fakes his?                                                         
Upon our gatesy: yed, if this sheal hath be scratch'd                                                                                                                                                       
His now fardelligent, and painty lips uportune                                                        
Solutated s appretition. This wise fink,                                                              
Which for sent fains him half cause 'tis truth:                                                       
If it were, unless for evil here yet then                                                             
I cannot not stable.

VOLUMBERLAND:                                                                                         
His only then; 'twere lost thy tears,                                                                 
Forgive the Duke's devis
