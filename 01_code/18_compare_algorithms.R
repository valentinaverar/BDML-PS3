############################################################
# PS3 | Equipo 03 | Algorithm Comparison
# Ejecutar desde la raíz del repo. No estima modelos ni modifica insumos.
############################################################
local({
  library(ggplot2)
  library(dplyr)
  dir.create("02_outputs/figures", recursive = TRUE, showWarnings = FALSE)
  dir.create("02_outputs/tables", recursive = TRUE, showWarnings = FALSE)
  train <- read.csv("00_data/raw/train.csv", fileEncoding = "UTF-8")
  test <- read.csv("00_data/raw/test.csv", fileEncoding = "UTF-8")
  kaggle <- read.csv("00_data/kaggle_scores.csv", fileEncoding = "UTF-8-BOM")
  registro <- read.csv("02_outputs/tables/registro_modelos.csv", fileEncoding = "UTF-8")
  importancia <- read.csv("02_outputs/tables/LGBM_vecinos_importancia.csv")
  seleccion <- read.csv("02_outputs/tables/seleccion_cv.csv", fileEncoding = "UTF-8")
  esp <- filter(registro, tipo_cv == "espacial")
  azul <- "#2F5FA8"; gris <- "#9DA8AE"; naranja <- "#D87642"; verde <- "#268D77"; tinta <- "#293B44"
  tema <- theme_minimal(base_size = 12) + theme(text = element_text(colour = tinta),
    plot.title = element_text(face = "bold", size = 13), panel.grid.minor = element_blank(),
    legend.position = "top", panel.grid.major.y = element_blank(), axis.title.y = element_blank())
  fmt <- function(x, d=1) formatC(x, digits=d, format="f", decimal.mark=",")
  guardar <- function(p, nombre, w=8, h=4.2) {
    ggsave(file.path("02_outputs/figures", paste0(nombre,".pdf")), p,
           width=w, height=h, device=if (capabilities("cairo")) cairo_pdf else "pdf")
  }
  public <- function(p) min(kaggle$puntaje_publico[grepl(p, kaggle$archivo)], na.rm=TRUE)/1e6
  espacial <- function(p) min(esp$mae[grepl(p,esp$archivo)],na.rm=TRUE)/1e6
  # 5: interacciones. No se imputan superficies para estas gráficas descriptivas.
  a <- train |> mutate(area=coalesce(surface_covered,surface_total)) |>
    filter(between(area,20,500),price>0) |>
    mutate(grupo=cut(area,c(20,60,90,130,200,500),include.lowest=TRUE,
                     labels=c("20–60","60–90","90–130","130–200","200–500")))
  celdas <- a |> group_by(property_type,grupo) |> summarise(med=median(price)/1e6,n=n(),.groups="drop") |> filter(n>=50)
  g <- ggplot(celdas,aes(grupo,property_type,fill=med)) + geom_tile(colour="white") +
    geom_text(aes(label=paste0(round(med),"\n(n=",n,")"),colour=med>850),size=4.2,fontface="bold") +
    scale_colour_manual(values=c("TRUE"="white","FALSE"=tinta),guide="none") +
    scale_fill_gradient(low="#E4EDF8",high=azul,name="Mediana\n(M COP)") +
    labs(title="El precio por tamaño difiere según el tipo de inmueble",x="Superficie observada (m²)",y=NULL)+tema+
    theme(panel.grid=element_blank(),legend.position="right")
  guardar(g,"compare_interacciones",8.1,3.65)
  # 6: forma funcional del precio por metro cuadrado.
  apt <- filter(a,property_type=="Apartamento") |> mutate(unit=price/area/1e6)
  cortes <- unique(quantile(apt$area,seq(0,1,.1)))
  deciles <- apt |> mutate(decil=cut(area,cortes,include.lowest=TRUE)) |> group_by(decil) |>
    summarise(area=median(area),unit=median(unit),.groups="drop")
  g <- ggplot(deciles,aes(area,unit))+geom_hline(yintercept=median(apt$unit),linetype=2,colour=gris)+
    geom_line(colour=azul,linewidth=1)+geom_point(colour=azul,size=3)+
    labs(title="El precio por m² no es constante",x="Superficie mediana de cada decil (m²)",y="Precio mediano por m² (M COP)",
      caption="Línea gris: mediana global")+tema+theme(axis.title.y=element_text(),panel.grid.major.y=element_line(colour="#E4E7E9"))
  guardar(g,"compare_forma")
  # 7: familia, mejor MAE espacial y puntaje Kaggle de la MISMA especificación.
  familias <- esp |> mutate(familia=case_when(grepl("^LM_",archivo)~"Regresión lineal",grepl("^EN_",archivo)~"Elastic Net",
    grepl("^CART_",archivo)~"CART",grepl("^RF_",archivo)~"Random Forest",grepl("^LGBM_",archivo)~"LightGBM",
    grepl("^NN_",archivo)~"Red neuronal",grepl("^BLEND_",archivo)~"Ensambles",TRUE~algoritmo)) |>
    group_by(familia) |> mutate(variantes=n_distinct(archivo)) |> slice_min(mae,n=1,with_ties=FALSE) |> ungroup() |>
    left_join(select(kaggle,archivo,puntaje_publico),by="archivo") |> transmute(familia,variantes,archivo,mae_cv=mae/1e6,mae_kaggle=puntaje_publico/1e6)
  write.csv(familias,"02_outputs/tables/compare_familias_cv.csv",row.names=FALSE)
  # 8: misma información (174 variables), misma CV espacial, sin híbridos ni vecinos.
  ids <- c("^LGBM_completo_lr_","^NN_","^RF_completo_mtry_","^EN_completo_","^LM_completo$","^CART_completo_cp_")
  df <- data.frame(modelo=c("LightGBM","Red neuronal","Random Forest","Elastic Net","Regresión lineal","CART"),
                   mae=vapply(ids,espacial,numeric(1))) |> arrange(mae)
  df$modelo <- factor(df$modelo,levels=rev(df$modelo))
  g <- ggplot(df,aes(mae,modelo))+geom_point(aes(colour=modelo=="LightGBM"),size=3.7)+
    geom_text(aes(label=fmt(mae,2)),hjust=-.25,size=3.6)+scale_colour_manual(values=c("TRUE"=azul,"FALSE"=gris),guide="none")+
    expand_limits(x=max(df$mae)+13)+labs(x="MAE en CV espacial (millones de COP)")+tema
  guardar(g,"compare_igualdad")
  # 10: objetivo total versus híbrido precio por m², comparado en los mismos folds.
  d <- data.frame(modelo=rep(c("CART","Random Forest","LightGBM"),2),
    objetivo=rep(c("Precio total","Híbrido m²"),each=3),mae=c(vapply(c("^CART_completo_cp_","^RF_completo_mtry_","^LGBM_completo_lr_"),espacial,numeric(1)),
    vapply(c("^CART_completo_m2_","^RF_completo_m2_","^LGBM_completo_m2_lr_"),espacial,numeric(1))))
  d$modelo <- factor(d$modelo,levels=rev(c("CART","Random Forest","LightGBM")))
  g <- ggplot(d,aes(mae,modelo))+geom_line(aes(group=modelo),colour=gris,linewidth=.7)+geom_point(aes(colour=objetivo),size=3.4)+
    geom_text(aes(label=fmt(mae),colour=objetivo),position=position_dodge(width=.5),vjust=-1,size=3.5)+
    scale_colour_manual(values=c("Precio total"=gris,"Híbrido m²"=azul))+labs(x="MAE CV espacial (M COP)",colour=NULL)+tema
  guardar(g,"compare_objetivo",7.6,3.6)
  # 11: tres mejores envíos públicos. Se empareja por nombre de archivo.
  top <- kaggle |> arrange(puntaje_publico) |> slice_head(n=3) |> left_join(esp,by="archivo",suffix=c("","_cv"))
  etiquetas <- c("BLEND vecinos + red","LightGBM + vecinos","BLEND m² + red")
  comparar <- bind_rows(data.frame(modelo=etiquetas,grupo="CV espacial",mae=top$mae/1e6),
    data.frame(modelo=etiquetas,grupo="10% similares a Chapinero",mae=top$mae_similares_chapinero/1e6),
    data.frame(modelo=etiquetas,grupo="Kaggle público",mae=top$puntaje_publico/1e6))
  comparar$modelo <- factor(comparar$modelo,levels=rev(etiquetas))
  pos <- position_dodge(width=.62)
  g <- ggplot(comparar,aes(mae,modelo,colour=grupo))+geom_point(position=pos,size=3)+
    geom_text(aes(label=fmt(mae)),position=pos,hjust=-.25,size=3.1)+
    scale_colour_manual(values=c("CV espacial"=gris,"10% similares a Chapinero"=azul,"Kaggle público"=naranja))+
    expand_limits(x=max(comparar$mae,na.rm=TRUE)+10)+labs(x="MAE (M COP)",colour=NULL)+tema
  guardar(g,"compare_tres",7.6,3.7)
  # Mapa descriptivo: rejilla angular de aproximadamente 400 m.
  rejilla <- train |> mutate(gx=floor(lon/.0036)*.0036+.0018,gy=floor(lat/.0036)*.0036+.0018) |>
    group_by(gx,gy) |> summarise(med=median(price)/1e6,n=n(),.groups="drop") |> filter(n>=5)
  g <- ggplot(rejilla,aes(gx,gy))+geom_tile(aes(fill=med),width=.0036,height=.0036)+
    geom_point(data=test,aes(lon,lat),inherit.aes=FALSE,colour=naranja,alpha=.22,size=.35)+
    scale_fill_gradient(low="#DDE8F8",high=azul,name="Mediana train\n(M COP)")+coord_equal()+
    labs(title="Precios observados y territorio a predecir",caption="Naranja: test sin precios")+theme_void(base_size=12)+
    theme(plot.title=element_text(face="bold"))
  guardar(g,"compare_mapa",6.8,5.3)
  # Opcional: mapa de errores REAL, si están disponibles las predicciones OOF.
  rutas <- file.path("03_temp",c("train_model.rds","oof_blend_vecinos_red.rds","oof_lgbm_vecinos.rds","oof_blend_red.rds"))
  salida_oof <- "02_outputs/figures/compare_errores_oof.pdf"
  if (all(file.exists(rutas))) {
    tm <- readRDS(rutas[1]); tm <- tm[,c("property_id","price","lat","lon","property_type")]
    oof <- lapply(rutas[-1],readRDS)
    for (i in seq_along(oof)) {
      x <- oof[[i]]; v <- grep("_espacial$",names(x),value=TRUE)
      stopifnot(length(v)==1L, !anyDuplicated(x$property_id))
      tm[[paste0("pred",i)]] <- x[[v]][match(tm$property_id,x$property_id)]
    }
    stopifnot(!anyNA(tm[c("pred1","pred2","pred3")]))
    tm <- tm |> mutate(gx=floor(lon/.009)*.009+.0045,gy=floor(lat/.009)*.009+.0045)
    errores <- bind_rows(lapply(seq_along(etiquetas),function(i) data.frame(tm[,c("property_id","property_type","price","gx","gy")],
                  modelo=etiquetas[i],err=abs(tm[[paste0("pred",i)]]-tm$price)/1e6)))
    zonas <- errores |> group_by(gx,gy,modelo) |> summarise(mae=mean(err),n=n(),.groups="drop") |> filter(n>=30)
    ganadores <- zonas |> group_by(gx,gy) |> slice_min(mae,n=1,with_ties=FALSE) |> ungroup()
    colores <- setNames(c(azul,naranja,verde),etiquetas)
    mapa <- ggplot(ganadores,aes(gx,gy,fill=modelo))+geom_tile(width=.009,height=.009)+coord_equal()+
      scale_fill_manual(values=colores)+labs(title="Menor MAE OOF por celda (aprox. 1 km)",fill=NULL)+theme_void()+theme(legend.position="bottom",plot.title=element_text(size=10))
    seg <- errores |> group_by(modelo,property_type) |> summarise(mae=mean(err),.groups="drop")
    graf <- ggplot(seg,aes(mae,property_type,colour=modelo))+geom_point(position=position_dodge(width=.55),size=3)+
      scale_colour_manual(values=colores)+labs(title="Error por tipo de inmueble",x="MAE espacial (M COP)",colour=NULL)+tema+theme(legend.position="bottom",plot.title=element_text(size=10))
    write.csv(seg,"02_outputs/tables/compare_oof_tipo.csv",row.names=FALSE)
    write.csv(zonas,"02_outputs/tables/compare_oof_zonas.csv",row.names=FALSE)
    grDevices::pdf(salida_oof,width=10,height=4.2)
    grid::grid.newpage(); lay <- grid::grid.layout(1,2); grid::pushViewport(grid::viewport(layout=lay))
    print(mapa,vp=grid::viewport(layout.pos.row=1,layout.pos.col=1)); print(graf,vp=grid::viewport(layout.pos.row=1,layout.pos.col=2)); grDevices::dev.off()
  } else {
    # No reutilizar un mapa de otra corrida si faltan los OOF de esta.
    unlink(salida_oof)
    message("Mapa OOF pendiente: faltan archivos locales de los tres modelos; se usa el mapa descriptivo.")
  }
  # 12: ganancia de la rama precio/m², separada por origen del predictor.
  imp <- filter(importancia,modelo=="m2") |> mutate(texto=grepl("^(raiz_|txt_)",Feature))
  dic <- c(vec_precio_m2="Precio/m² vecinos",surface_covered_completa="Área cubierta",surface_total_completa="Área total",vec_precio="Precio vecinos",dist_centro_empleo_m="Dist. empleo",lat="Latitud",pct_zona_verde_500m="Zona verde",dist_via_principal_m="Dist. vía principal",raiz_balcon="Balcón",txt_parqueaderos="Parqueaderos",txt_piso="Piso",raiz_abiert="Abierto",raiz_terraz="Terraza",raiz_gimnasi="Gimnasio",raiz_bbq="BBQ",raiz_remodel="Remodelación")
  for (txt in c(FALSE,TRUE)) {
    d <- filter(imp,texto==txt) |> slice_max(Gain,n=8,with_ties=FALSE) |> mutate(etiqueta=coalesce(unname(dic[Feature]),Feature),etiqueta=reorder(etiqueta,Gain))
    g <- ggplot(d,aes(100*Gain,etiqueta))+geom_col(fill=if(txt) verde else azul,width=.65)+
      geom_text(aes(label=fmt(100*Gain)),hjust=-.2,size=3.5)+expand_limits(x=max(d$Gain)*120)+labs(x="Ganancia relativa (%)")+tema
    guardar(g,if(txt) "compare_texto" else "compare_importancia",5.4,4.4)
  }
  # 13: secuencia de envíos; no atribuir causalmente la mejora.
  p <- c("^LM_base$","^LM_completo$","^LGBM_completo_lr_","^LGBM_completo_m2_lr_","^LGBM_completo_m2_vecinos_k10_lr_","^BLEND_vecinos_red_")
  d <- data.frame(paso=c("MCO\nbase","MCO\ncompleto","LightGBM\ncompleto","Precio\npor m²","+ vecinos","+ red"),mae=vapply(p,public,numeric(1)))
  d$paso <- factor(d$paso,levels=d$paso)
  g <- ggplot(d,aes(paso,mae,group=1))+geom_line(colour=gris)+geom_point(colour=azul,size=3.6)+
    geom_text(aes(label=fmt(mae)),vjust=-1,size=3.8)+expand_limits(y=max(d$mae)+20)+labs(x=NULL,y="MAE público (M COP)")+tema+theme(axis.title.y=element_text(),panel.grid.major.y=element_line(colour="#E4E7E9"))
  guardar(g,"compare_factores",8.3,4.1)
  # 15: comparación de ocho esquemas de validación.
  seleccion$esquema <- reorder(seleccion$esquema,-seleccion$brecha_absoluta_pct)
  g <- ggplot(seleccion,aes(brecha_absoluta_pct,esquema))+geom_point(aes(colour=elegido),size=3.5)+
    geom_text(aes(label=fmt(brecha_absoluta_pct,2)),hjust=-.2,size=3.3)+scale_colour_manual(values=c("TRUE"=azul,"FALSE"=gris),guide="none")+
    expand_limits(x=max(seleccion$brecha_absoluta_pct)+3)+labs(x="Brecha absoluta promedio (%)")+tema
  guardar(g,"compare_cv_esquemas")
  message("Comparación generada; no se reentrenaron modelos.")
})
